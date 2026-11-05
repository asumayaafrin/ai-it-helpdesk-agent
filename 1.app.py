"""Streamlit UI for AI IT Helpdesk Agent.   Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from engine import DEFAULT_THRESHOLDS, HelpdeskEngine, infer_priority, ollama_available
from tickets import PRIORITIES, STATUSES, TicketDB

st.set_page_config(page_title="AI IT Helpdesk Agent", page_icon="🛠️", layout="wide")


@st.cache_resource(show_spinner="Loading knowledge base and models...")
def get_engine() -> HelpdeskEngine:
    return HelpdeskEngine(use_embeddings=True)


@st.cache_resource
def get_db() -> TicketDB:
    return TicketDB()


engine, db = get_engine(), get_db()
ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("pending", None)  # last answered query waiting for Solved / Not solved feedback

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🛠️ AI IT Helpdesk Agent")
    user_name = st.text_input("Your name", value="", placeholder="e.g. Sumaya")

    st.subheader("Settings")
    modes = ["hybrid", "embedding", "tfidf"] if engine.retriever.embeddings_available else ["tfidf"]
    mode = st.selectbox("Search method", modes,
                        help="TF-IDF = keyword match. Embedding = meaning match. Hybrid = both.")
    default_thr = DEFAULT_THRESHOLDS[engine.retriever.resolve_mode(mode)]
    threshold = st.slider("Confidence threshold", 0.05, 0.9, float(default_thr), 0.01,
                          help="Below this score the bot does not guess; it creates a ticket.")
    use_llm = st.toggle("Rewrite with Llama (Ollama)", value=True)
    show_debug = st.toggle("Show match details", value=True)

    st.divider()
    st.caption("System status")
    st.write("🟢 Ollama running" if ollama_available() else "🟠 Ollama offline (showing KB steps directly)")
    if engine.retriever.embeddings_available:
        st.write("🟢 Sentence embeddings loaded")
    else:
        st.write("🟠 Embeddings unavailable, using TF-IDF only")
        if engine.retriever.embed_error:
            st.caption(engine.retriever.embed_error[:160])

    if st.button("Clear chat"):
        ss.messages, ss.pending = [], None
        st.rerun()

tab_chat, tab_tickets, tab_stats = st.tabs(["💬 Chat", "🎫 Tickets", "📊 Analytics"])

# ------------------------------------------------------------------- chat
with tab_chat:
    st.header("How can I help you today?")
    st.caption("Describe your problem: Wi-Fi, VPN, printer, password, software, hardware, email...")

    for m in ss.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("debug") and show_debug:
                st.caption(m["debug"])

    query = st.chat_input("Type your IT issue...")
    if query:
        ss.messages.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            ans = engine.answer(query, mode=mode, threshold=threshold, use_llm=use_llm)

        best = ans.best
        debug = (f"Matched: {best.entry['title']} | score {best.score:.2f} (threshold {ans.threshold:.2f}) | "
                 f"method: {ans.mode}" + (" | LLM rewrite" if ans.used_llm else " | KB steps"))

        if ans.confident:
            iid = db.log_interaction(user_name, query, best.entry["id"], best.entry["category"],
                                     best.score, ans.mode, "answered")
            ss.pending = {"interaction_id": iid, "query": query, "category": best.entry["category"],
                          "priority": best.entry["priority"], "score": best.score}
            reply = ans.text
        else:
            tid = db.create_ticket(user_name, query, "Unclassified", infer_priority(query),
                                   "low_confidence", best.score)
            db.log_interaction(user_name, query, None, None, best.score, ans.mode, "ticket")
            ss.pending = None
            reply = (f"I'm not confident I can solve this one, so I created **ticket #{tid}** "
                     f"for the IT team. They will contact you soon.\n\n"
                     f"_Closest topic I found: {best.entry['title']}._")
        ss.messages.append({"role": "assistant", "content": reply, "debug": debug})
        st.rerun()

    # feedback for the latest answer
    if ss.pending:
        st.write("**Did this solve your problem?**")
        c1, c2, _ = st.columns([1, 1.6, 6])
        if c1.button("✅ Solved"):
            db.set_outcome(ss.pending["interaction_id"], "solved")
            ss.messages.append({"role": "assistant", "content": "Great, glad it's fixed! 🎉"})
            ss.pending = None
            st.rerun()
        if c2.button("❌ Not solved"):
            p = ss.pending
            db.set_outcome(p["interaction_id"], "not_solved")
            prio = "high" if p["priority"] == "high" else infer_priority(p["query"], "medium")
            tid = db.create_ticket(user_name, p["query"], p["category"], prio, "user_feedback", p["score"])
            ss.messages.append({"role": "assistant", "content":
                                f"Sorry about that. I created **ticket #{tid}** ({p['category']}, {prio} priority) "
                                "so a human can take over."})
            ss.pending = None
            st.rerun()

# ---------------------------------------------------------------- tickets
with tab_tickets:
    st.header("Ticket dashboard")
    f1, f2 = st.columns([1, 3])
    status_filter = f1.selectbox("Filter by status", ["All"] + STATUSES)
    tickets = db.list_tickets(status_filter)

    if tickets:
        df = pd.DataFrame(tickets)[["id", "created_at", "user_name", "issue", "category",
                                    "priority", "status", "source"]]
        st.dataframe(df, width="stretch", hide_index=True)

        st.subheader("Update a ticket")
        u1, u2, u3 = st.columns([1, 1.5, 1])
        tid = u1.selectbox("Ticket ID", [t["id"] for t in tickets])
        new_status = u2.selectbox("New status", STATUSES)
        if u3.button("Update", width="stretch"):
            db.update_status(tid, new_status)
            st.success(f"Ticket #{tid} is now {new_status}")
            st.rerun()
    else:
        st.info("No tickets yet.")

# -------------------------------------------------------------- analytics
with tab_stats:
    st.header("Analytics")
    s = db.stats()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total queries", s["total_queries"])
    k2.metric("Handled without ticket",
              "-" if s["auto_handled_rate"] is None else f"{s['auto_handled_rate']:.0%}")
    k3.metric("User-confirmed resolution",
              "-" if s["resolution_rate"] is None else f"{s['resolution_rate']:.0%}")
    k4.metric("Open tickets", s["open_tickets"])

    if s["tickets_by_category"]:
        st.subheader("Tickets by category")
        st.bar_chart(pd.Series(s["tickets_by_category"]))
    else:
        st.info("Analytics will appear once people start using the bot.")
