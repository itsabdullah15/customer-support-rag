import os
import json
import logging
from pathlib import Path
from typing import List
import requests
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "3"

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
BASE_DIR = Path(__file__).parent
DOCS_PATH = BASE_DIR / "documents.json"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
PINECONE_INDEX_NAME = "vector-search-index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
TOP_K = 3
MAX_CONTEXT_LENGTH = 4000


# === UTILS ===

def get_pinecone_client() -> pinecone.Pinecone:
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found.")
    return pinecone.Pinecone(api_key=api_key)

def get_groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found.")
    return api_key

def load_documents(doc_path: Path) -> List[str]:
    if not doc_path.exists():
        raise FileNotFoundError(f"{doc_path} not found.")
    with doc_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def truncate_documents(docs: List[str], max_length: int) -> str:
    context = ""
    for doc in docs:
        if len(context) + len(doc) + 2 > max_length:
            remaining = max_length - len(context) - 2
            context += doc[:remaining] + "\n\n"
            break
        context += doc + "\n\n"
    return context.strip()

def search_documents(index: pinecone.Index, query_vec: List[float], all_docs: List[str], k: int = 3) -> List[str]:
    try:
        results = index.query(vector=query_vec, top_k=k)
        return [all_docs[int(match["id"])] for match in results.get("matches", []) if int(match["id"]) < len(all_docs)]
    except Exception as e:
        raise RuntimeError(f"Document search failed: {e}")

def query_groq_model(prompt: str, api_key: str, model_name: str = LLM_MODEL_NAME) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant answering based on the context."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 512
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        raise RuntimeError(f"Groq API call failed: {e}")


# === LANGCHAIN COMPONENTS ===

# Prompt Template
prompt_template = PromptTemplate.from_template(
    """You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}"""
)

# LangChain Runnables
def embed_query_fn(query: str) -> List[float]:
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model.encode(query, convert_to_numpy=True).tolist()

embed_chain = RunnableLambda(embed_query_fn)


# === MAIN CHAIN ===

def prepare_query(query: str, docs: List[str]) -> str:
    query_vector = embed_chain.invoke(query)
    index = get_pinecone_client().Index(PINECONE_INDEX_NAME)
    relevant_docs = search_documents(index, query_vector, docs, TOP_K)

    if not relevant_docs:
        return "No relevant documents found."

    context = truncate_documents(relevant_docs, MAX_CONTEXT_LENGTH)
    prompt = prompt_template.invoke({
        "context": context,
        "question": query
    })

    # Ensure prompt is a string
    return prompt.get_text() if hasattr(prompt, "get_text") else str(prompt)



def generate_answer(prompt: str) -> str:
    groq_key = get_groq_api_key()
    return query_groq_model(prompt, groq_key)


def main(query: str) -> str:
    try:
        logger.info("Loading resources...")
        docs = load_documents(DOCS_PATH)
        prompt = prepare_query(query, docs)

        logger.info("Calling Groq LLM...")
        answer = generate_answer(prompt)

        print("Answer:", answer)
        return answer

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        return f"Error: {e}"


if __name__ == "__main__":
    # Example query
    sample_query = "How is the PM of India?"
    main(sample_query)
