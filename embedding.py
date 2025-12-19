"""
Модуль для создания эмбеддингов из текстового контекста тренировок
"""
import os
import sqlite3
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
TABLE = "TrainingContext"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Кеш для модели
_model = None

print(f"[INFO] DB_PATH в embedding.py: {DB_PATH}")


def get_model():
    """Получает модель эмбеддингов (ленивая загрузка)"""
    global _model
    if _model is None:
        print(f"Загружаю модель:   {EMBEDDING_MODEL}")
        os.environ['HF_HOME'] = r"D:\huggingface_cache"
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def create_embeddings():
    """Создает эмбеддинги для всех текстов контекста, которые еще не имеют эмбеддингов"""
    
    # Проверяем что БД существует
    if not os.path.exists(DB_PATH):
        print(f"✗ БД не найдена: {DB_PATH}")
        raise FileNotFoundError(f"БД не найдена: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    try:
        # Проверяем что таблица существует
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{TABLE}'")
        if not cur.fetchone():
            print(f"✗ Таблица {TABLE} не найдена в {DB_PATH}")
            raise ValueError(f"Таблица {TABLE} не найдена")
        
        print(f"✓ Таблица {TABLE} найдена")
        
        # Получаем записи без эмбеддингов
        cur.execute(f"""
            SELECT id, context_text 
            FROM {TABLE} 
            WHERE embedding IS NULL
            LIMIT 1000
        """)
        rows = cur.fetchall()
        
        print(f"Найдено {len(rows)} записей без эмбеддингов")
        
        if not rows:
            print(f"✓ Все записи в {TABLE} уже имеют эмбеддинги")
            return
        
        print(f"Создаю эмбеддинги для {len(rows)} записей...")
        model = get_model()
        
        for idx, (row_id, text) in enumerate(rows, 1):
            try:
                # Создаем эмбеддинг
                embedding = model.encode(text)
                embedding_bytes = pickle.dumps(embedding)
                
                # Сохраняем в БД
                cur.execute(f"""
                    UPDATE {TABLE}
                    SET embedding = ?   
                    WHERE id = ?  
                """, (embedding_bytes, row_id))
                
                if idx % 10 == 0:
                    print(f"  {idx}/{len(rows)} обработано...")
            
            except Exception as e:
                print(f"  ✗ Ошибка при обработке записи {row_id}: {e}")
        
        conn.commit()
        print(f"✓ Эмбеддинги успешно созданы для {len(rows)} записей")
        
    except Exception as e:
        print(f"✗ Ошибка при создании эмбеддингов: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        raise
    finally:
        conn.close()


def get_embeddings_for_search(query: str) -> np.ndarray:
    """Создает эмбеддинг для поискового запроса"""
    model = get_model()
    embedding = model.encode(query)
    return embedding
