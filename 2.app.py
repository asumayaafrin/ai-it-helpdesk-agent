import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
from datetime import datetime

OLLAMA_MODEL = "llama3.2:1b"

kb = pd.read_csv("it_issues.csv")
vectorizer = TfidfVectorizer()
issue_vectors = vectorizer.fit_transform(kb["issue"])

DB_FILE = "tickets.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            timestamp TEXT,
            issue TEXT,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def call_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def retrieve_relevant_issue(user_query, top_k=1):
    query_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vector, issue_vectors).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    idx = top_indices[0]
    return {
        "issue": kb.iloc[idx]["issue"],
        "category": kb.iloc[idx]["category"],
        "solution": kb.iloc[idx]["solution_steps"],
        "confidence": round(similarities[idx], 2)
    }

def create_ticket(issue_description):
    ticket_id = f"TCK-{int(datetime.now().timestamp())}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO tickets (ticket_id, timestamp, issue, status) VALUES (?, ?, ?, ?)",
        (ticket_id, timestamp, issue_description, "Escalated to IT Team")
    )
    conn.commit()
    conn.close()
    return ticket_id

def get_all_tickets():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM tickets", conn)
    conn.close()
    return df

def ai_helpdesk_agent(user_query):
    retrieved = retrieve_relevant_issue(user_query)
    if retrieved["confidence"] < 0.3:
        return {"response": None, "resolved": False, "retrieved": retrieved}
    prompt = f"""You are a professional IT helpdesk agent.
User issue: \"{user_query}\"
KB Issue: {retrieved['issue']}
Category: {retrieved['category']}
Solution steps: {retrieved['solution']}

Respond friendly and professional. Diagnose briefly, then give numbered solution steps."""
    answer = call_ollama(prompt)
    return {"response": answer, "resolved": True, "retrieved": retrieved}

st.set_page_config(page_title="AI IT Helpdesk Agent", page_icon="🛠️")
st.title("🛠️ AI IT Helpdesk Agent")
st.caption("Agent + RAG + Tools + SQL | Powered by Ollama (local LLM) | Describe your IT issue below")

if "history" not in st.session_state:
    st.session_state.history = []

user_query = st.chat_input("Describe your IT issue...")

if user_query:
    result = ai_helpdesk_agent(user_query)
    st.session_state.history.append(("user", user_query))
    if result["resolved"]:
        st.session_state.history.append(("agent", result["response"]))
    else:
        ticket_id = create_ticket(user_query)
        msg = f"I couldn't find a confident match. I've escalated this — Ticket ID: **{ticket_id}**"
        st.session_state.history.append(("agent", msg))

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.write(msg)

st.sidebar.header("📋 Raised Tickets (from SQL database)")
tickets_df = get_all_tickets()
if not tickets_df.empty:
    st.sidebar.dataframe(tickets_df)
else:
    st.sidebar.write("No tickets yet.")
