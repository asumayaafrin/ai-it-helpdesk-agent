"""Core brain of AI IT Helpdesk Agent.

Pipeline:
    user query -> retrieve best KB issue (TF-IDF / sentence embeddings / hybrid)
               -> confidence check against a threshold
               -> high confidence: Ollama (Llama 3.2 1B) rewrites the KB steps (grounded, no new steps)
                  Ollama down: fall back to the KB steps directly
               -> low confidence: caller creates a ticket
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
KB_PATH = BASE_DIR / "data" / "kb.json"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# tfidf: tuned with evaluate.py (threshold sweep, zero wrong answers on the test set).
# embedding / hybrid: STARTING GUESSES. Run `python evaluate.py` where the model can download
# (Colab) and copy the printed best thresholds here. Re-run after you change the KB.
DEFAULT_THRESHOLDS = {"tfidf": 0.30, "embedding": 0.45, "hybrid": 0.35}

HIGH_PRIORITY_WORDS = {
    "urgent", "asap", "immediately", "critical", "down", "outage", "production",
    "cannot work", "unable to work", "client", "deadline", "stolen", "hacked", "virus",
}


def load_kb(path: Path = KB_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def infer_priority(query: str, fallback: str = "medium") -> str:
    q = clean(query)
    if any(w in q for w in HIGH_PRIORITY_WORDS):
        return "high"
    return fallback


@dataclass
class Match:
    entry: dict
    score: float
    tfidf: float = 0.0
    embedding: float | None = None


@dataclass
class Answer:
    confident: bool
    best: Match | None
    candidates: list[Match] = field(default_factory=list)
    text: str = ""
    used_llm: bool = False
    mode: str = "tfidf"
    threshold: float = 0.0


class Retriever:
    """Finds the closest KB issue for a query.

    TF-IDF  : keyword overlap. Fast, no model download, but does not understand meaning.
    Embedding: sentence-transformers (all-MiniLM-L6-v2). Understands "net slow" ~ "wifi is slow".
    Hybrid  : weighted mix of both (best of both worlds).
    """

    def __init__(self, kb: list[dict], use_embeddings: bool = True, alpha: float = 0.5):
        self.kb = kb
        self.alpha = alpha  # weight of embedding score in hybrid mode

        # --- TF-IDF: one document per issue (title x2 + all symptom phrasings) ---
        self.docs = [clean(" ".join([e["title"]] * 2 + e["symptoms"])) for e in kb]
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(self.docs)

        # --- Embeddings: one vector per phrase, issue score = best phrase ---
        self.embeddings_available = False
        self.embed_error = None
        self._phrase_owner: list[int] = []
        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(EMBED_MODEL)
                phrases = []
                for i, e in enumerate(kb):
                    for p in [e["title"]] + e["symptoms"]:
                        phrases.append(clean(p))
                        self._phrase_owner.append(i)
                self._phrase_vecs = self._model.encode(phrases, normalize_embeddings=True)
                self._owner = np.array(self._phrase_owner)
                self.embeddings_available = True
            except Exception as exc:  # no package, no internet to download model, etc.
                self.embed_error = f"{type(exc).__name__}: {exc}"

    # -------------------------------------------------------------- scoring
    def _tfidf_scores(self, query: str) -> np.ndarray:
        q = self.tfidf.transform([clean(query)])
        return cosine_similarity(q, self.tfidf_matrix)[0]

    def _embedding_scores(self, query: str) -> np.ndarray:
        qv = self._model.encode([clean(query)], normalize_embeddings=True)[0]
        sims = self._phrase_vecs @ qv
        scores = np.full(len(self.kb), -1.0)
        for idx in range(len(self.kb)):
            scores[idx] = sims[self._owner == idx].max()
        return scores

    def resolve_mode(self, mode: str) -> str:
        if mode in ("embedding", "hybrid") and not self.embeddings_available:
            return "tfidf"
        return mode

    def search(self, query: str, mode: str = "hybrid", top_k: int = 3) -> tuple[list[Match], str]:
        mode = self.resolve_mode(mode)
        t = self._tfidf_scores(query)
        e = self._embedding_scores(query) if self.embeddings_available and mode != "tfidf" else None

        if mode == "tfidf":
            final = t
        elif mode == "embedding":
            final = e
        else:
            final = self.alpha * e + (1 - self.alpha) * t

        order = np.argsort(-final)[:top_k]
        matches = [
            Match(self.kb[i], float(final[i]), float(t[i]), None if e is None else float(e[i]))
            for i in order
        ]
        return matches, mode


# ------------------------------------------------------------------ answers
def format_steps(entry: dict) -> str:
    lines = [f"**{entry['title']}**", ""]
    lines += [f"{n}. {s}" for n, s in enumerate(entry["steps"], 1)]
    return "\n".join(lines)


def ollama_available(timeout: float = 1.5) -> bool:
    try:
        return requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout).ok
    except requests.RequestException:
        return False


def rewrite_with_llm(query: str, entry: dict, timeout: float = 60) -> str | None:
    """Ask Llama to phrase the KB steps for this user. Grounded: it may not invent steps."""
    steps = "\n".join(f"{n}. {s}" for n, s in enumerate(entry["steps"], 1))
    prompt = (
        "You are a polite IT helpdesk assistant.\n"
        "Use ONLY the approved steps below. Do not add, remove or invent steps, commands or settings.\n"
        "Reply with a one-line empathetic opening, then the steps as a numbered list, in simple English.\n\n"
        f"User problem: {query}\n\nApproved steps for '{entry['title']}':\n{steps}\n"
    )
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.2, "num_predict": 400}},
            timeout=timeout,
        )
        r.raise_for_status()
        text = r.json().get("response", "").strip()
        return text or None
    except (requests.RequestException, ValueError):
        return None


class HelpdeskEngine:
    def __init__(self, use_embeddings: bool = True):
        self.kb = load_kb()
        self.retriever = Retriever(self.kb, use_embeddings=use_embeddings)

    def answer(self, query: str, mode: str = "hybrid", threshold: float | None = None,
               use_llm: bool = True) -> Answer:
        matches, used_mode = self.retriever.search(query, mode=mode)
        best = matches[0]
        thr = DEFAULT_THRESHOLDS[used_mode] if threshold is None else threshold

        if best.score < thr:
            return Answer(False, best, matches, mode=used_mode, threshold=thr)

        text, used_llm = None, False
        if use_llm:
            text = rewrite_with_llm(query, best.entry)
            used_llm = text is not None
        if not text:
            text = format_steps(best.entry)
        return Answer(True, best, matches, text=text, used_llm=used_llm, mode=used_mode, threshold=thr)
