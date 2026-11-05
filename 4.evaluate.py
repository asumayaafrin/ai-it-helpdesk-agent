"""Measure retrieval quality and pick a confidence threshold.

    python evaluate.py              # all available methods
    python evaluate.py --no-embed   # TF-IDF only (no model download)

In-scope queries  (expected_id = number): should be answered with the RIGHT issue.
Out-of-scope queries (expected_id = null): should NOT be answered; they should become a ticket.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from engine import HelpdeskEngine

TEST_PATH = Path(__file__).parent / "data" / "test_queries.json"


def evaluate(engine: HelpdeskEngine, tests: list[dict], mode: str):
    rows = []
    for t in tests:
        matches, used = engine.retriever.search(t["query"], mode=mode, top_k=3)
        rows.append({
            "expected": t["expected_id"],
            "top1": matches[0].entry["id"],
            "top3": [m.entry["id"] for m in matches],
            "score": matches[0].score,
            "query": t["query"],
        })
    return rows, used


def sweep(rows, thresholds):
    ins = [r for r in rows if r["expected"] is not None]
    outs = [r for r in rows if r["expected"] is None]
    table = []
    for thr in thresholds:
        correct = sum(1 for r in ins if r["score"] >= thr and r["top1"] == r["expected"])
        wrong_answer = sum(1 for r in ins if r["score"] >= thr and r["top1"] != r["expected"])
        escalated = sum(1 for r in ins if r["score"] < thr)
        out_ticket = sum(1 for r in outs if r["score"] < thr)
        out_guess = len(outs) - out_ticket
        # A wrong answer is worse than an extra ticket (the user wastes time on a bad fix),
        # so utility = correct answers - 2 x wrong answers (in-scope wrong + out-of-scope guessed).
        bad = wrong_answer + out_guess
        utility = correct - 2 * bad
        table.append((thr, correct, wrong_answer, escalated, out_ticket, out_guess, utility))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-embed", action="store_true")
    args = ap.parse_args()

    tests = json.loads(TEST_PATH.read_text(encoding="utf-8"))
    engine = HelpdeskEngine(use_embeddings=not args.no_embed)
    n_in = sum(1 for t in tests if t["expected_id"] is not None)
    n_out = len(tests) - n_in
    print(f"Test set: {len(tests)} queries ({n_in} in-scope, {n_out} out-of-scope)")
    print(f"KB size : {len(engine.kb)} issues")
    if not args.no_embed and not engine.retriever.embeddings_available:
        print(f"[!] Embeddings unavailable -> {engine.retriever.embed_error}")

    modes = ["tfidf"]
    if engine.retriever.embeddings_available:
        modes += ["embedding", "hybrid"]

    summary = {}
    for mode in modes:
        rows, _ = evaluate(engine, tests, mode)
        ins = [r for r in rows if r["expected"] is not None]
        top1 = np.mean([r["top1"] == r["expected"] for r in ins])
        top3 = np.mean([r["expected"] in r["top3"] for r in ins])
        print(f"\n=== {mode.upper()} ===")
        print(f"Retrieval accuracy (in-scope, ignoring threshold): top-1 {top1:.0%} | top-3 {top3:.0%}")

        print("\n(in-scope: correct / wrong answer / escalated to ticket)  (out-of-scope: ticketed / guessed)")
        print("thr   correct  wrong  escalated | ticketed  guessed | utility  precision")
        best = None
        for thr, correct, wrong, esc, o_t, o_g, util in sweep(rows, np.arange(0.05, 0.95, 0.05)):
            answered = correct + wrong + o_g
            prec = correct / answered if answered else float("nan")
            print(f"{thr:.2f}  {correct:7d}  {wrong:5d}  {esc:9d} | {o_t:8d}  {o_g:7d} | {util:7d}  {prec:8.0%}")
            if best is None or util > best[1]:
                best = (thr, util, correct, wrong + o_g, esc)
        print(f"-> best threshold on this test set: {best[0]:.2f} "
              f"({best[2]} correct, {best[3]} wrong answers, {best[4]} in-scope escalated)")
        summary[mode] = (top1, top3, best)

        misses = [r for r in ins if r["top1"] != r["expected"]]
        if misses:
            print("\nWrong top-1 matches:")
            for r in misses:
                print(f"  expected #{r['expected']} got #{r['top1']} (score {r['score']:.2f}): {r['query']}")

    print("\n=== SUMMARY ===")
    for mode, (t1, t3, (thr, _, correct, wrong, esc)) in summary.items():
        print(f"{mode:10s} top-1 {t1:.0%} | top-3 {t3:.0%} | threshold {thr:.2f} -> "
              f"{correct}/{n_in} auto-resolved correctly, {wrong} wrong answers, {esc} escalated")


if __name__ == "__main__":
    main()
