import os
import sqlite3
import numpy as np
from embed_model import get_embed_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
TABLE = "VectorMetadata"


def ensure_table_and_column():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(f"PRAGMA table_info({TABLE})")
    cols = [c[1] for c in cur.fetchall()]

    if "embedding" not in cols:
        cur.execute(f"ALTER TABLE {TABLE} ADD COLUMN embedding BLOB")
        conn.commit()

    conn.close()


def create_embeddings(batch_size=64):
    ensure_table_and_column()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        f"SELECT index_id, text FROM {TABLE} WHERE embedding IS NULL"
    )
    rows = cur.fetchall()

    if not rows:
        print("Эмбеддинги уже существуют.")
        conn.close()
        return

    model = get_embed_model()

    print(f"Создание эмбеддингов для {len(rows)} записей...")
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = [r[1] or "" for r in batch]

        vecs = model.encode(
            texts,
            normalize_embeddings=True
        ).astype(np.float32)

        for (index_id, _), vec in zip(batch, vecs):
            cur.execute(
                f"UPDATE {TABLE} SET embedding=? WHERE index_id=?",
                (vec.tobytes(), index_id)
            )

        conn.commit()
        print(f"Обработано {i + len(batch)}/{len(rows)}")

    conn.close()
    print("Эмбеддинги созданы.")
