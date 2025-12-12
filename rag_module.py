# rag_module.py
import os
import sqlite3
import pickle
import numpy as np
import faiss

# LLM optional
try:
    from llama_cpp import Llama
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
RAG_DIR = os.path.join(BASE_DIR, "rag")
FAISS_PATH = os.path.join(RAG_DIR, "vector_index.bin")
META_PATH = os.path.join(RAG_DIR, "vector_index.bin.meta")
MODEL_GGUF = os.path.join(BASE_DIR, "models", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")  

os.makedirs(RAG_DIR, exist_ok=True)
CACHE_DIR = r"D:\huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

# embedding model for queries
query_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

TOP_K = 6
MAX_TOKENS = 512
N_THREADS = 6
HISTORY_KEEP = 6  # keep last N turns (user+assistant pairs) when building prompt

# ---------- DB helpers ----------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS rag_cache(
                        query TEXT PRIMARY KEY,
                        answer TEXT
                    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS rag_history(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        role TEXT,       -- 'user' or 'assistant'
                        text TEXT,
                        ts DATETIME DEFAULT CURRENT_TIMESTAMP
                    )""")
    conn.commit()
    return conn

# ---------- FAISS ----------
def load_faiss():
    if not os.path.exists(FAISS_PATH) or not os.path.exists(META_PATH):
        raise FileNotFoundError("FAISS index or meta not found. Run build_faiss.py")
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def semantic_search(query_text, top_k=TOP_K):
    index, meta = load_faiss()
    q_emb = np.array(query_model.encode([query_text]), dtype="float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    seen_texts = set()
    for idx in I[0]:
        if idx < len(meta):
            text = meta[idx].get("text","")
            if text and text not in seen_texts:
                seen_texts.add(text)
                results.append(meta[idx])
    return results

# ---------- History handling ----------
def append_history(role: str, text: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO rag_history(role, text) VALUES (?, ?)", (role, text))
    conn.commit()
    conn.close()

def get_recent_history(max_pairs=HISTORY_KEEP):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, text FROM rag_history ORDER BY id DESC LIMIT ?", (max_pairs*2,))
    rows = cur.fetchall()[::-1]  # reverse back to chronological order
    conn.close()
    cleaned = []
    prev = None
    for role, text in rows:
        if text.strip() == "":
            continue
        if prev is None or text.strip() != prev:
            cleaned.append((role, text.strip()))
            prev = text.strip()
    return cleaned[-(max_pairs*2):]

def build_history_block():
    hist = get_recent_history(HISTORY_KEEP)
    if not hist:
        return ""
    out_lines = []
    for role, text in hist:
        tag = "Пользователь" if role == "user" else "Ассистент"
        out_lines.append(f"{tag}: {text}")
    return "\n".join(out_lines)

# ---------- Prompt building ----------
def build_prompt(user_query: str, retrieved: list):
    history_block = build_history_block()
    seen = set()
    ctx_lines = []
    for item in retrieved:
        t = item.get("text","").strip()
        if not t or t in seen:
            continue
        if len(t) > 800:
            t = t[:800] + "..."
        ctx_lines.append(t)
    context = "\n\n".join(ctx_lines)

    prompt_sections = []
    prompt_sections.append("<|begin_of_text|>")
    if history_block:
        prompt_sections.append("История диалога (последние ходы):\n" + history_block)
    if context:
        prompt_sections.append("Релевантный контекст из базы:\n" + context)
    prompt_sections.append(f"Вопрос пользователя: {user_query}")
    prompt_sections.append("<|end_of_text|>")
    return "\n\n".join(prompt_sections)

# ---------- LLM call ----------
def call_llm(prompt: str) -> str:
    if LLM_AVAILABLE and os.path.exists(MODEL_GGUF):
        try:
            llm = Llama(model_path=MODEL_GGUF, n_threads=N_THREADS, n_ctx=2048)
            resp = llm(prompt, max_tokens=MAX_TOKENS)
            text = resp.get("choices", [{}])[0].get("text", "").strip()
            if text:
                return text
        except Exception as e:
            return f"(LLM error) модель выдала ошибку: {e}\n\nСформированный ответ на основе контекста:\n\n{prompt}"
    out = "LLM недоступен — возвращаю сводку по контексту и инструкцию:\n\n"
    out += "Контекст:\n" + (prompt[:4000] + ("..." if len(prompt)>4000 else ""))
    out += "\n\nИнструкция: Сформируй программу тренировок на основе приведённого контекста и запроса."
    return out

# ---------- caching ----------
def get_cached(query: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT answer FROM rag_cache WHERE query = ?", (query,))
    r = cur.fetchone()
    conn.close()
    return r[0] if r else None

def cache_answer(query: str, answer: str):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("INSERT OR REPLACE INTO rag_cache(query, answer) VALUES (?, ?)", (query, answer))
        conn.commit()
    except Exception:
        pass
    conn.close()

# ---------- main API ----------
def answer_question(user_query: str, top_k=TOP_K):
    user_query = user_query.strip()
    if user_query == "":
        return "Пустой запрос."

    hist = get_recent_history(1)
    if hist and hist[-1][0] == "user" and hist[-1][1].strip().lower() == user_query.lower():
        return "Этот запрос уже был получен как последний — повторный ответ пропущен."

    cached = get_cached(user_query)
    if cached:
        append_history("user", user_query)
        append_history("assistant", cached)
        return cached

    retrieved = semantic_search(user_query, top_k=top_k)
    prompt = build_prompt(user_query, retrieved)
    answer = call_llm(prompt)
    cache_answer(user_query, answer)
    append_history("user", user_query)
    append_history("assistant", answer)

    return answer

# quick test
if __name__ == "__main__":
    print(answer_question("Сделай программу на 3 дня в неделю, цель — набор массы"))
