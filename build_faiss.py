# build_faiss.py
import os
import sqlite3
import pickle
import numpy as np

# try import faiss; helpful error if missing
try:
    import faiss
except Exception as e:
    raise ImportError("faiss not installed (pip install faiss-cpu). Full error: " + str(e))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
TABLE = "VectorMetadata"
RAG_DIR = os.path.join(BASE_DIR, "rag")
FAISS_PATH = os.path.join(RAG_DIR, "vector_index.bin")
META_PATH = os.path.join(RAG_DIR, "vector_index.bin.meta")

os.makedirs(RAG_DIR, exist_ok=True)

def build_index():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # select rows which have embedding
    cur.execute(f"SELECT index_id, text, embedding FROM {TABLE} WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    conn.close()
    if not rows:
        raise RuntimeError("No embeddings found. Run embedding.create_embeddings() first.")

    ids = []
    meta = []
    vectors = []
    for index_id, text, emb in rows:
        try:
            vec = np.frombuffer(emb, dtype=np.float32)
        except Exception:
            continue
        vectors.append(vec)
        ids.append(index_id)
        meta.append({"index_id": index_id, "text": text})

    embeddings = np.vstack(vectors).astype("float32")
    # normalize for cosine similarity using inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"FAISS index built ({index.ntotal} vectors) -> {FAISS_PATH}")
    print(f"Metadata saved -> {META_PATH}")

if __name__ == "__main__":
    build_index()
