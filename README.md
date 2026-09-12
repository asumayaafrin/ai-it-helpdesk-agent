🛠️ AI IT Helpdesk Agent

An Agentic AI system that diagnoses common IT issues, reasons through a fix using an LLM, and automatically raises a support ticket when it can't resolve a problem — built as part of an IBM Internship (Agentic AI) project.

Capabilities demonstrated: Agent + RAG + Tools + SQL

📌 Problem Statement

IT support teams spend a large share of their time on repetitive, low-complexity issues — password resets, VPN problems, printer errors, slow systems — while employees wait for basic help that could be resolved instantly. This project automates that first line of support with an autonomous AI agent.

💡 How It Works

The agent follows a 4-stage pipeline for every user query:

Retrieve (RAG) — The query is matched against a knowledge base of 15 common IT issues using TF-IDF + cosine similarity.
Reason (Agent) — The retrieved issue and solution are passed to an LLM, which generates a clear, step-by-step diagnosis and fix.
Decide & Act (Tools) — If the match confidence is too low, the agent calls a tool to automatically create a support ticket instead of guessing.
Persist (SQL) — Every escalated ticket is stored in a SQLite database with a ticket ID, timestamp, issue, and status, and shown live on a dashboard.
✨ Key Features
Conversational chat interface built with Streamlit
RAG-based retrieval over a 15-issue IT knowledge base
LLM-powered natural language diagnosis
Automatic ticket creation and escalation for unresolved issues
SQL (SQLite) database for persistent ticket storage
Live sidebar dashboard showing all raised tickets
Two interchangeable LLM backends — Groq (cloud API) and Ollama (local model) — demonstrating a model-agnostic agent design
🧰 Tech Stack
Frontend + Backend — Streamlit
Agent / LLM Reasoning — Groq API (openai/gpt-oss-120b) and Ollama (llama3.2:1b, local)
RAG / Retrieval — scikit-learn (TF-IDF + Cosine Similarity)
Data Handling — Pandas
Database — SQLite
Environment — Google Colab
Deployment — Cloudflare Tunnel
📂 Repository Structure
IT_Helpdesk_Agent_with_Groq.ipynb — Full notebook, Groq (cloud LLM) version
IT_Helpdesk_Agent_with_Ollama.ipynb — Full notebook, Ollama (local LLM) version
app.py — Streamlit application
it_issues.csv — RAG knowledge base (15 common IT issues)
tickets.db — SQLite database of escalated tickets
AI_IT_Helpdesk_Agent_Report.pdf — Full project report
README.md — This file
🚀 How to Run
Open either notebook in Google Colab
Run every cell in order, top to bottom
The last cell prints a public Cloudflare URL — open it in a browser to use the live app
Type an IT issue in the chat box (e.g. "wifi is not connecting") to see the agent diagnose and respond, or an unrelated issue (e.g. "my chair is broken") to see it auto-escalate and raise a ticket

Note: Google Colab sessions disconnect after inactivity or on a new day — simply re-run all cells from the top to redeploy.

📊 Results
Correctly diagnosed known issues (e.g. WiFi, printer problems) with clear, LLM-generated step-by-step solutions
Correctly escalated unrecognized issues, generating valid tickets (e.g. TCK-1789200602) stored in and retrievable from the SQL database
Verified working end-to-end with both a cloud LLM (Groq) and a locally-hosted LLM (Ollama)
🔮 Future Enhancements
Add conversational memory for multi-turn troubleshooting
Expand the knowledge base with real-world ticket data
Integrate with a production ticketing system (e.g. Zoho Desk, ServiceNow)
Permanent cloud deployment (Streamlit Community Cloud / Render)
Voice input/output support
👤 Author

Sumaya Afrin A B.E. CSE (AI & ML), Mangayarkarasi College of Engineering, Anna University IBM Internship — Agentic AI

Built as part of an IBM Internship project on Agentic AI (Use Case: AI IT Helpdesk Agent).
