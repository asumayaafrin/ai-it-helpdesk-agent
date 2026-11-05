# ai-it-helpdesk-agent

AI IT Helpdesk Agent (Agentic AI) Project.

# AI IT Helpdesk Agent

This repository contains the AI IT Helpdesk Agent project on Agentic AI.

## Project

An Agentic AI system that diagnoses common IT issues (Wi-Fi, VPN, printer, password, software, hardware, email, security), presents a fix as clear step-by-step instructions using an LLM, and automatically raises a support ticket when it is not confident it can resolve the problem.

## Capabilities Demonstrated

- Agent
- RAG (Retrieval-Augmented Generation)
- Tools (ticket creation backed by SQL)
- SQL

## Technologies Used

- Python
- Streamlit
- Ollama (Llama 3.2 1B)
- scikit-learn (TF-IDF + Cosine Similarity)
- sentence-transformers (optional semantic search, falls back to TF-IDF)
- Pandas
- SQLite
- Google Colab
- Cloudflare Tunnel

## How It Works

```
User query
   |
   v
Retriever: TF-IDF (keywords) / sentence embeddings (meaning) / hybrid
   |
   v
Confidence >= threshold?
   |-- YES -> Llama 3.2 1B (Ollama) rewrites the approved knowledge-base steps
   |          (it cannot invent steps; falls back to the KB steps if Ollama is offline)
   |          User confirms "Solved" or "Not solved" (Not solved creates a ticket)
   |-- NO  -> Ticket created in SQLite (status: Open) for the IT team
```

## Features

- Knowledge base of 45 IT issues across 8 categories (`data/kb.json`)
- Three search methods: TF-IDF, sentence embeddings, or a hybrid of both
- Confidence threshold: low-confidence queries become tickets instead of wrong answers
- Grounded LLM answers: Llama only rephrases the approved steps
- Ticket system: user, category, priority, status (Open / In Progress / Resolved / Closed), source and confidence
- Streamlit chat with "Solved / Not solved" feedback, a ticket dashboard and an analytics tab

## Evaluation

`evaluate.py` runs a 61-query test set (51 in-scope, 10 out-of-scope) and sweeps the confidence threshold.

TF-IDF baseline results:

- Top-1 retrieval accuracy: 90%, top-3: 98%
- At threshold 0.30: 34 of 51 in-scope queries answered correctly, 0 wrong answers, 17 escalated to tickets; all 10 out-of-scope queries became tickets
- Lower thresholds answer more queries but begin giving wrong answers

The test set was written together with the knowledge base, so scores are optimistic. Add real user queries to `data/test_queries.json` for a fairer result.

## Run Locally

```bash
pip install -r requirements.txt
ollama pull llama3.2:1b      # keep `ollama serve` running
streamlit run app.py
python evaluate.py           # accuracy and threshold sweep
```

## Run on Google Colab

Open `run_on_colab.ipynb`, upload the project zip when asked, and run the cells in order. The last cell prints a public Cloudflare Tunnel link to the app.

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: chat, ticket dashboard, analytics |
| `engine.py` | Retriever (TF-IDF / embeddings / hybrid), threshold logic, Ollama call |
| `tickets.py` | SQLite tickets and interaction log |
| `evaluate.py` | Accuracy and threshold sweep |
| `data/kb.json` | Knowledge base (45 issues) |
| `data/test_queries.json` | Evaluation queries |
| `run_on_colab.ipynb` | Colab runner |

## Limitations and Future Work

- The knowledge base is hand-written; there is no admin screen to add issues
- No login or per-user history
- SQLite suits a demo; a production system would use PostgreSQL
- Llama 3.2 1B is small, but it only rephrases approved steps
- Colab with Cloudflare Tunnel is a demo setup, not a permanent deployment

## Student Details

**Name:** A.Sumaya Afrin

**College:** Mangayarkarasi College of Engineering

**Department:** B.E. CSE (AI & ML)

**University:** Anna University

## Report

[View Full Internship Report](AI_IT_Helpdesk_Agent_Report.pdf)
