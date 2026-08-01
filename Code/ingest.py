import os
import glob
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "kb")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Simple chunking: split by paragraph, keep chunks under ~500 chars
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def embed_text(text, task_type="RETRIEVAL_DOCUMENT"):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"task_type": task_type, "output_dimensionality": 768}
    )
    return result.embeddings[0].values

DOMAIN_MAP = {
    "VPN Access Request Guide.txt": "account",
    "Jira Access Request Process.txt": "account",
    "Password Reset Instructions.txt": "account",
    "New Employee Onboarding Checklist.txt": "account",
    "Common Platform Errors and Fixes.txt": "platform",
    "Requesting New Software Installation.txt": "platform",
}

def ingest_kb():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(name="helpdesk_kb")

    doc_id = 0
    for filepath in glob.glob(os.path.join(KB_DIR, "*.txt")):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk)
            collection.add(
                ids=[f"{filename}_{i}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename, "domain": DOMAIN_MAP.get(filename, "platform")}]
            )
            doc_id += 1
            print(f"Indexed chunk {doc_id}: {filename} part {i}")

    print(f"\nDone. Total chunks indexed: {doc_id}")
    return doc_id

if __name__ == "__main__":
    ingest_kb()