import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "policies"
VECTOR_DIR = BASE_DIR / "vector_store"
EVAL_DIR = BASE_DIR / "evaluation"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "420"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "180"))

DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "10"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "6"))
FINAL_TOP_N = int(os.getenv("FINAL_TOP_N", "4"))

LOW_SUPPORT_THRESHOLD = float(os.getenv("LOW_SUPPORT_THRESHOLD", "0.56"))
HIGH_SUPPORT_THRESHOLD = float(os.getenv("HIGH_SUPPORT_THRESHOLD", "0.78"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai")

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "90"))
