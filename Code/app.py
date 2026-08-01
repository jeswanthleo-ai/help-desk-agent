import os
import streamlit as st

# Streamlit Cloud injects secrets via st.secrets, not .env/os.environ. Bridge
# them so the rest of the codebase can keep using os.getenv() unchanged.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except FileNotFoundError:
    pass  # no secrets.toml locally — .env via load_dotenv() covers this case

from graph import graph
from email_escalation import send_escalation_email
from sub_agents import collection
from ingest import ingest_kb

st.set_page_config(page_title="Help Desk Agent", page_icon="🎧")

if collection.count() == 0:
    with st.spinner("Setting up knowledge base for the first time..."):
        ingest_kb()
st.title("🎧 Help Desk Agent")
st.caption("Agentic RAG-powered support — ask about access, accounts, or platform issues")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke({"query": prompt, "escalate": False})

            if result["escalate"]:
                send_escalation_email(prompt, result["agent_used"])
                response = (
                    "I couldn't find a confident answer in our knowledge base, "
                    "so I've escalated your question to the support team. "
                    "They'll follow up with you directly."
                )
            else:
                response = result["answer"]

            st.markdown(response)
            st.caption(f"Routed to: {result['agent_used']}")

    st.session_state.messages.append({"role": "assistant", "content": response})