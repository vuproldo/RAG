# run_rug.py
import os
import sqlite3
from embedding import create_embeddings
from build_faiss import build_index
from rag_module import answer_question

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
TABLE = "VectorMetadata"
RAG_DIR = os.path.join(BASE_DIR, "rag")
FAISS_PATH = os.path.join(RAG_DIR, "vector_index.bin")

# ensure hf cache folder exists (rag_module also sets, but safe here)
os.makedirs(r"D:\huggingface_cache", exist_ok=True)

def ensure_embeddings_and_faiss():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({TABLE})")
    cols = [c[1] for c in cur.fetchall()]

    # add embedding column if missing
    if "embedding" not in cols:
        print("Колонка 'embedding' отсутствует — создаём её.")
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN embedding BLOB;")
        conn.commit()

    # check if embeddings exist
    cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE embedding IS NOT NULL")
    cnt = cur.fetchone()[0]
    if cnt == 0:
        print("Эмбеддинги не найдены — создаём эмбеддинги...")
        create_embeddings()
    else:
        print(f"Эмбеддинги обнаружены: {cnt}")

    conn.close()

    # build faiss if not exists
    if not os.path.exists(FAISS_PATH):
        print("FAISS индекс не найден — строим...")
        build_index()
    else:
        print("FAISS индекс найден.")

def main():
    print("\n--- Подготовка RAG ---")
    ensure_embeddings_and_faiss()

    print("\n=== RAG Chat (persistent history). Введите 'exit' для выхода ===")
    while True:
        q = input("\nВаш вопрос: ").strip()
        if q.lower() in ("exit", "quit", "выход"):
            print("Выход.")
            break
        try:
            ans = answer_question(q)
            print("\n=== Ответ ===\n")
            print(ans)
            print("\n---\n")
        except Exception as e:
            print("Ошибка при обработке запроса:", e)

if __name__ == "__main__":
    main()
