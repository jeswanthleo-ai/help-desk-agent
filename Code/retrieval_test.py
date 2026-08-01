import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="helpdesk_kb")

def embed_query(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 768}
    )
    return result.embeddings[0].values

query = "How do I get access to Jira?"
query_embedding = embed_query(query)

results = collection.query(query_embeddings=[query_embedding], n_results=3)

for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    print(f"\n--- Source: {meta['source']} | Distance: {dist:.4f} ---")
    print(doc)