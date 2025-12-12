# embedding.py
import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
TABLE = "VectorMetadata"


CACHE_DIR = r"D:\huggingface_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)

# load embedding model
print(">>> Loading embedding model (nomic-ai/nomic-embed-text-v1.5)...")
embed_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
print(">>> Embedding model loaded.")

def ensure_table_and_column():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    # ensure table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (TABLE,))
    if not cur.fetchone():
        conn.close()
        raise FileNotFoundError(f"Table {TABLE} not found in DB {DB_PATH}")
    # ensure embedding column exists
    cur.execute(f"PRAGMA table_info({TABLE});")
    cols = [r[1] for r in cur.fetchall()]
    if "embedding" not in cols:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN embedding BLOB;")
        conn.commit()
    conn.close()

def create_embeddings(batch_size: int = 64):
    """
    Create embeddings for rows in VectorMetadata.text where embedding is NULL.
    Saves embeddings as raw bytes (float32).
    """
    ensure_table_and_column()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT index_id, text FROM {TABLE} WHERE embedding IS NULL OR embedding = ''")
    rows = cur.fetchall()
    if not rows:
        print("No new rows to embed.")
        conn.close()
        return

    print(f"Creating embeddings for {len(rows)} rows...")
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        texts = [r[1] if r[1] is not None else "" for r in batch]
        vecs = embed_model.encode(texts, show_progress_bar=False)
        vecs = np.array(vecs, dtype=np.float32)
        for (index_id, _), vec in zip(batch, vecs):
            cur.execute(f"UPDATE {TABLE} SET embedding = ? WHERE index_id = ?", (vec.tobytes(), index_id))
        conn.commit()
        print(f"  processed {min(i+batch_size, len(rows))}/{len(rows)}")
    conn.close()
    print("Embeddings created.")
    

if __name__ == "__main__":
    create_embeddings()
