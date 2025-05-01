# Load FAISS + Groq LLM → build QA chain
import os
import json
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# Configuration
BASE_DIR = Path(__file__).parent
INDEX_PATH = BASE_DIR / "faiss_index.index"
DOCS_PATH = BASE_DIR / "documents.json"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
K = 3  # Number of documents to retrieve

def load_resources(index_path: str, docs_path: str) -> tuple[faiss.Index, List[str]]:
    """Load FAISS index and document texts with error handling."""
    try:
        index = faiss.read_index(str(index_path))
        with open(docs_path, "r", encoding="utf-8") as f:
            texts = json.load(f)
        return index, texts
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Resource not found: {e}")
    except Exception as e:
        raise Exception(f"Failed to load resources: {e}")

def get_groq_api_key() -> str:
    """Retrieve Groq API key with validation."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return api_key

def search_documents(index: faiss.Index, texts: List[str], query: str, model: SentenceTransformer, k: int = 3) -> List[str]:
    """Embed query and retrieve top-k documents."""
    query_vector = model.encode(query, convert_to_numpy=True)
    D, I = index.search(query_vector[np.newaxis, :], k)  # Ensure correct shape
    return [texts[i] for i in I[0] if i < len(texts)]  # Guard against invalid indices

def query_groq(context: str, query: str, api_key: str, model_name: str = MODEL_NAME) -> str:
    """Query Groq API with context and question."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant answering based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        "temperature": 0.5,
        "max_tokens": 512
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for bad status codes
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        raise Exception(f"Groq API request failed: {e}")

def truncate_context(documents: List[str], max_length: int = 4000) -> str:
    """Truncate documents to fit within token limits."""
    context = ""
    for doc in documents:
        if len(context) + len(doc) + 2 > max_length:
            remaining = max_length - len(context) - 2
            context += doc[:remaining] + "\n\n"
            break
        context += doc + "\n\n"
    return context.strip()

def main(query: str) -> None:
    """Main function to run the QA pipeline."""
    try:
        # Load resources
        index, texts = load_resources(INDEX_PATH, DOCS_PATH)
        api_key = get_groq_api_key()

        # Initialize embedding model
        model = SentenceTransformer(EMBEDDING_MODEL)

        # Search for relevant documents
        retrieved_docs = search_documents(index, texts, query, model, K)
        if not retrieved_docs:
            print("No relevant documents found.")
            return

        # Truncate and format context
        context = truncate_context(retrieved_docs)

        # Query Groq LLM
        answer = query_groq(context, query, api_key)
        print("Answer:", answer)
        return answer

    except Exception as e:
        print("Error:", str(e))

