"""
Главный скрипт для запуска RAG системы с контекстом из TrainingDiary. db
"""
import os
import sqlite3
from generate_training_context import generate_training_context
from embedding import create_embeddings
from build_faiss import build_index_from_db, get_index_info
from rag_module import answer_question

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiary.db")
RAG_DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
FAISS_PATH = os. path.join(BASE_DIR, "rag", "vector_index.bin")

# Кеш для модели HuggingFace
os.makedirs(r"D:\huggingface_cache", exist_ok=True)

print(f"[INFO] TRAINING_DB_PATH: {TRAINING_DB_PATH}")
print(f"[INFO] RAG_DB_PATH: {RAG_DB_PATH}")
print(f"[INFO] FAISS_PATH: {FAISS_PATH}")


def ensure_rag_database():
    """Создает БД для RAG если её нет"""
    os.makedirs(os.path. dirname(RAG_DB_PATH), exist_ok=True)
    if not os.path.exists(RAG_DB_PATH):
        conn = sqlite3.connect(RAG_DB_PATH)
        conn.close()
        print(f"✓ Создана БД:  {RAG_DB_PATH}")


def ensure_rag_system():
    """Проверяет и инициализирует RAG систему"""
    print("\n--- Инициализация RAG системы ---\n")
    
    # 1. Проверяем исходную БД тренировок
    if not os.path.exists(TRAINING_DB_PATH):
        print(f"✗ Не найдена БД тренировок: {TRAINING_DB_PATH}")
        print(f"  Проверьте что файл существует по адресу выше")
        return False
    print(f"✓ БД тренировок найдена: {TRAINING_DB_PATH}")
    
    # 2. Создаем БД RAG если нужно
    ensure_rag_database()
    
    # 3. Генерируем контекст из тренировок
    print("\n[Шаг 1/3] Генерирую контекст из тренировок...")
    context_count = generate_training_context()
    if context_count == 0:
        print("⚠ Контекст не был создан")
        return False
    
    # 4. Создаем эмбеддинги
    print("\n[Шаг 2/3] Создаю эмбеддинги...")
    try:
        create_embeddings()
    except Exception as e:
        print(f"✗ Ошибка при создании эмбеддингов: {e}")
        return False
    
    # 5. Строим FAISS индекс
    print("\n[Шаг 3/3] Строю FAISS индекс...")
    try:
        build_index_from_db()
    except Exception as e:  
        print(f"✗ Ошибка при построении индекса: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n[Проверка FAISS индекса]")
    if not os.path.exists(FAISS_PATH):
        print(f"✗ FAISS индекс не найден: {FAISS_PATH}")
        return False
    
    print(get_index_info())
    print("✓ FAISS индекс найден и готов к использованию!\n")
    
    print("✓ RAG система успешно инициализирована!\n")
    return True


def main():
    """Главная функция"""
    # Инициализируем RAG систему
    if not ensure_rag_system():
        print("✗ Ошибка инициализации RAG системы")
        return
    
    # Запускаем интерактивный чат
    print("=== RAG Chat - Вопросы о ваших тренировках ===")
    print("Введите 'exit', 'quit' или 'выход' для выхода\n")
    
    while True:
        try:
            question = input("\nВаш вопрос: ").strip()
            
            if question.lower() in ("exit", "quit", "выход", ""):
                if question == "":  
                    continue
                print("До свидания!")
                break
            
            print("\n[Обработка запроса...]")
            answer = answer_question(question)
            print("\n=== Ответ ===\n")
            print(answer)
            print("\n---\n")
        
        except KeyboardInterrupt:  
            print("\n\nПрограмма прервана пользователем.")
            break
        except Exception as e:
            print(f"Ошибка при обработке запроса: {e}")


if __name__ == "__main__":
    main()
