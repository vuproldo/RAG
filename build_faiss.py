"""
Модуль для построения FAISS индекса из эмбеддингов тренировочных данных
"""
import os
import sqlite3
import pickle
import numpy as np
import faiss
from typing import List, Tuple

BASE_DIR = os.path. dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
RAG_DIR = os.path.join(BASE_DIR, "rag")
FAISS_PATH = os. path.join(RAG_DIR, "vector_index.bin")
METADATA_PATH = os.path.join(RAG_DIR, "metadata.pkl")
TABLE = "TrainingContext"


def build_index_from_db():
    """Строит FAISS индекс из эмбеддингов в TrainingDiaryVector.db"""
    os.makedirs(RAG_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    try:
        # Получаем все эмбеддинги и метаданные
        cur.execute(f"""
            SELECT id, embedding, context_text, user_id
            FROM {TABLE}
            WHERE embedding IS NOT NULL
            ORDER BY id
        """)
        rows = cur.fetchall()
        
        if not rows:
            print(f"✗ Нет эмбеддингов в таблице {TABLE}")
            return
        
        print(f"Строю FAISS индекс из {len(rows)} векторов...")
        
        # Распаковываем эмбеддинги
        vectors = []
        metadata = []
        
        for row in rows:
            try:
                embedding = pickle.loads(row['embedding'])
                vectors.append(embedding)
                metadata.append({
                    'id': row['id'],
                    'user_id': row['user_id'],
                    'text': row['context_text'][: 500]  # Сохраняем первые 500 символов
                })
            except Exception as e: 
                print(f"  ⚠ Ошибка при обработке записи {row['id']}: {e}")
        
        if not vectors:
            print(f"✗ Не удалось распаковать эмбеддинги")
            return
        
        vectors = np.array(vectors).astype('float32')
        print(f"  Векторы подготовлены: {vectors.shape}")
        
        # Создаем FAISS индекс
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        # Сохраняем индекс
        faiss.write_index(index, FAISS_PATH)
        print(f"✓ Индекс сохранен: {FAISS_PATH}")
        
        # Сохраняем метаданные
        with open(METADATA_PATH, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ Метаданные сохранены:  {METADATA_PATH}")
        
    except Exception as e:
        print(f"✗ Ошибка при построении индекса: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        conn. close()


def get_similar_contexts(query_embedding:  np.ndarray, k: int = 5) -> List[Tuple[str, float, int]]:
    """
    Получает k похожих контекстов для запроса, используя FAISS индекс
    """
    
    if not os.path.exists(FAISS_PATH):
        raise FileNotFoundError(f"FAISS индекс не найден: {FAISS_PATH}")
    
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Метаданные не найдены:  {METADATA_PATH}")
    
    try:
        # Загружаем существующий FAISS индекс
        print(f"[DEBUG] Загружаю индекс:  {FAISS_PATH}")
        index = faiss.read_index(FAISS_PATH)
        print(f"[DEBUG] Индекс загружен.  Количество векторов: {index.ntotal}")
        
        # Загружаем метаданные
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"[DEBUG] Загружено метаданных: {len(metadata)}")
        
        # Поиск похожих векторов
        query_vector = query_embedding. astype('float32').reshape(1, -1)
        distances, indices = index.search(query_vector, min(k, len(metadata)))
        
        print(f"[DEBUG] Найдено похожих записей: {len(indices[0])}")
        
        # Формируем результаты
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata):
                meta = metadata[idx]
                results.append((
                    meta['text'],
                    float(distances[0][i]),
                    meta['user_id']
                ))
                print(f"[DEBUG] Результат {i+1}: расстояние={distances[0][i]:.4f}, user_id={meta['user_id']}")
        
        return results
    
    except Exception as e: 
        print(f"✗ Ошибка при поиске контекстов: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_index_info():
    """Возвращает информацию об индексе"""
    if not os.path.exists(FAISS_PATH):
        return "Индекс не найден"
    
    try:
        index = faiss.read_index(FAISS_PATH)
        conn = sqlite3.connect(DB_PATH)
        cur = conn. cursor()
        cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE embedding IS NOT NULL")
        db_count = cur.fetchone()[0]
        conn.close()
        
        return f"""
FAISS Индекс информация:
- Путь:  {FAISS_PATH}
- Векторов в индексе: {index.ntotal}
- Записей в БД: {db_count}
- Размерность: {index. d}
"""
    except Exception as e: 
        return f"Ошибка при получении информации: {e}"
