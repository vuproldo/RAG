import os
import sqlite3
import pickle
import numpy as np
import faiss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
TABLE = "VectorMetadata"

RAG_DIR = os.path.join(BASE_DIR, "rag")
FAISS_PATH = os.path.join(RAG_DIR, "vector_index.bin")
META_PATH = os.path.join(RAG_DIR, "vector_index.bin.meta")
os.makedirs(RAG_DIR, exist_ok=True)


def build_index():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        f"SELECT index_id, text, embedding FROM {TABLE} "
        "WHERE embedding IS NOT NULL"
    )
    rows = cur.fetchall()
    conn.close()

    vectors = []
    meta = []
    dim = None

    for index_id, text, emb in rows:
        if not text or not emb:
            continue

        vec = np.frombuffer(emb, dtype=np.float32)
        if dim is None:
            dim = vec.shape[0]

        if vec.shape[0] != dim:
            continue

        vectors.append(vec)
        meta.append({"index_id": index_id, "text": text})

    if not vectors:
        raise RuntimeError("Нет валидных эмбеддингов")

    embeddings = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"FAISS index built: {index.ntotal}")


if __name__ == "__main__":
    build_index()
