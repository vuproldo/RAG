import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAG_DIR = os.path.join(BASE_DIR, "rag")
FAISS_PATH = os.path.join(RAG_DIR, "vector_index.bin")
META_PATH = os.path.join(RAG_DIR, "vector_index.bin.meta")

MODEL_GGUF = os.path.join(
    BASE_DIR,
    "models",
    "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)

CACHE_DIR = r"D:\huggingface_cache"
MODEL_BASE_DIR = os.path.join(
    CACHE_DIR,
    "models--nomic-ai--nomic-embed-text-v1.5",
    "snapshots"
)

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

TOP_K = 6
MAX_TOKENS = 1024
N_THREADS = 6

_query_model = None
_faiss_index = None
_faiss_meta = None
_llm = None


def _resolve_model_path():
    snaps = os.listdir(MODEL_BASE_DIR)
    if not snaps:
        raise RuntimeError("No HF snapshots found")
    return os.path.join(MODEL_BASE_DIR, snaps[0])


def get_query_model():
    global _query_model
    if _query_model is None:
        print(">>> Loading embedding model (OFFLINE)")
        _query_model = SentenceTransformer(
            _resolve_model_path(),
            trust_remote_code=True
        )
    return _query_model


def load_faiss():
    global _faiss_index, _faiss_meta
    if _faiss_index is None:
        print(">>> Loading FAISS index")
        _faiss_index = faiss.read_index(FAISS_PATH)
        with open(META_PATH, "rb") as f:
            _faiss_meta = pickle.load(f)
    return _faiss_index, _faiss_meta


def semantic_search(query):
    index, meta = load_faiss()
    model = get_query_model()

    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    _, indices = index.search(q_emb, TOP_K)

    results = []
    seen = set()

    for idx in indices[0]:
        if idx < len(meta):
            text = meta[idx]["text"]
            if text not in seen:
                seen.add(text)
                results.append(meta[idx])

    return results


def build_prompt(query, retrieved):
    context = "\n".join(f"- {r['text']}" for r in retrieved)
    return (
        "<|system|>\n"
        "Ты сертифицированный персональный фитнес-тренер.\n"
        "Используй ТОЛЬКО упражнения из контекста.\n"
        "<|end_of_system|>\n\n"
        "Контекст:\n"
        f"{context}\n\n"
        "<|user|>\n"
        f"{query}\n"
        "<|end|>"
    )


def get_llm():
    global _llm
    if _llm is None:
        print(">>> Loading LLM")
        _llm = Llama(
            model_path=MODEL_GGUF,
            n_ctx=2048,
            n_threads=N_THREADS
        )
    return _llm


def answer_question(query):
    retrieved = semantic_search(query)
    if not retrieved:
        return "Ничего не найдено."

    llm = get_llm()
    prompt = build_prompt(query, retrieved)

    response = llm(prompt, max_tokens=MAX_TOKENS)

    if isinstance(response, dict):
        text = response["choices"][0]["text"]
    else:
        text = response[0]["text"]

    return text.strip()
