import os
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import chromadb

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options=types.HttpOptions(
        timeout=15000,  # 15s per attempt
        retry_options=types.HttpRetryOptions(attempts=2, initial_delay=1, max_delay=5),
    ),
)

CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="helpdesk_kb")

ANSWER_PROMPT = """You are a helpdesk support agent. Answer the user's question
using ONLY the context below. If the context does not contain enough information
to answer confidently, respond with exactly: "NOT_FOUND"

Context:
{context}

User question: {query}

Answer:
"""

RELEVANCE_THRESHOLD = 0.8  # lower distance = more relevant; tune after testing


def embed_query(text: str):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 768}
    )
    return result.embeddings[0].values


def _retrieve_and_answer(query: str, agent_name: str, domain: str):
    """Shared logic — retrieval is now filtered to the agent's own domain."""
    try:
        query_embedding = embed_query(query)
    except (errors.APIError, httpx.TimeoutException, httpx.ConnectError):
        return {
            "answer": None,
            "escalate": True,
            "agent": agent_name,
            "retrieved_sources": [],
        }

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"domain": domain}  # <-- this is the actual domain separation
    )

    docs = results["documents"][0]
    distances = results["distances"][0] if results["distances"] else []
    sources = [m["source"] for m in results["metadatas"][0]]

    if not docs or min(distances) > RELEVANCE_THRESHOLD:
        return {
            "answer": None,
            "escalate": True,
            "agent": agent_name,
            "retrieved_sources": sources,
        }

    context = "\n\n".join(docs)
    try:
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=ANSWER_PROMPT.format(context=context, query=query)
        )
    except (errors.APIError, httpx.TimeoutException, httpx.ConnectError):
        return {
            "answer": None,
            "escalate": True,
            "agent": agent_name,
            "retrieved_sources": sources,
        }
    answer_text = response.text.strip()

    if answer_text == "NOT_FOUND":
        return {
            "answer": None,
            "escalate": True,
            "agent": agent_name,
            "retrieved_sources": sources,
        }

    return {
        "answer": answer_text,
        "escalate": False,
        "agent": agent_name,
        "retrieved_sources": sources,
    }


def platform_support_agent(query: str):
    """Handles technical errors, how-tos, troubleshooting. Searches only 'platform' domain docs."""
    return _retrieve_and_answer(query, agent_name="platform_support_agent", domain="platform")


def account_access_agent(query: str):
    """Handles access requests, permissions, onboarding, passwords. Searches only 'account' domain docs."""
    return _retrieve_and_answer(query, agent_name="account_access_agent", domain="account")


if __name__ == "__main__":
    print("--- Account Agent test ---")
    print(account_access_agent("How do I request Jira access?"))

    print("\n--- Platform Agent test ---")
    print(platform_support_agent("I'm getting a 403 error"))

    print("\n--- Cross-domain check (should escalate, wrong agent for this question) ---")
    print(account_access_agent("I'm getting a 403 error"))