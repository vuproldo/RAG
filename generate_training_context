"""
Модуль для генерации текстового контекста из TrainingDiary. db
Создает тренировочные записи для встраивания в RAG систему
"""
import os
import sqlite3
from datetime import datetime, timedelta
from typing import List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DB_PATH = os.path.join(BASE_DIR, "Database", "TrainingDiary.db")
RAG_DB_PATH = os. path.join(BASE_DIR, "Database", "TrainingDiaryVector.db")
CONTEXT_TABLE = "TrainingContext"


def ensure_context_table():
    """Создает таблицу TrainingContext если её нет"""
    conn = sqlite3.connect(RAG_DB_PATH)
    cur = conn.cursor()
    
    try:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {CONTEXT_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                context_text TEXT NOT NULL,
                source_table TEXT,
                source_id INTEGER,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        print(f"✓ Таблица {CONTEXT_TABLE} проверена/создана")
    except Exception as e:
        print(f"✗ Ошибка при создании таблицы:   {e}")
    finally:
        conn.close()


def clear_old_context():
    """Очищает старые контексты перед регенерацией"""
    conn = sqlite3.connect(RAG_DB_PATH)
    cur = conn.cursor()
    
    try:
        cur.execute(f"DELETE FROM {CONTEXT_TABLE}")
        conn.commit()
        print(f"✓ Старые контексты удалены")
    except Exception as e:
        print(f"⚠ Не удалось очистить старые контексты: {e}")
    finally:
        conn. close()


def get_training_day_name(training_conn, training_day_id):
    """Получает имя тренировочного дня"""
    try:
        cur = training_conn.cursor()
        # Пытаемся получить имя из связанной таблицы
        cur.execute("""
            SELECT name FROM ProgramTrainingDay 
            WHERE training_day_id = ?  
            LIMIT 1
        """, (training_day_id,))
        result = cur.fetchone()
        return result[0] if result else f"Тренировка {training_day_id}"
    except: 
        return f"Тренировка {training_day_id}"


def generate_training_context() -> int:
    """
    Генерирует текстовый контекст из TrainingDiary.db
    Возвращает количество созданных записей контекста
    """
    # 1. Убеждаемся, что таблица существует
    ensure_context_table()
    
    # 2. Очищаем старые контексты
    clear_old_context()
    
    # Проверяем исходную БД
    if not os.path.exists(TRAINING_DB_PATH):
        print(f"✗ Не найдена БД тренировок: {TRAINING_DB_PATH}")
        return 0
    
    training_conn = sqlite3.connect(TRAINING_DB_PATH)
    training_conn.row_factory = sqlite3.Row
    training_cur = training_conn.cursor()
    
    # Подключаемся к RAG БД
    rag_conn = sqlite3.connect(RAG_DB_PATH)
    rag_cur = rag_conn.cursor()
    
    context_count = 0
    
    try:
        # Получаем пользователей
        training_cur.execute("SELECT id, name, weight, height FROM Users")
        users = training_cur.fetchall()
        
        if not users:
            print("⚠ Пользователи не найдены в БД тренировок")
            return 0
        
        print(f"Найдено {len(users)} пользователей")
        
        for user in users:
            user_id = user['id']
            user_name = user['name']
            print(f"  Обработка пользователя: {user_name} (ID: {user_id})")
            
            # 1. Контекст из WORKOUT (все записи, не только последние 30 дней)
            training_cur.execute("""
                SELECT 
                    w.id,
                    w.  date,
                    w.duration,
                    w.total_volume,
                    w.training_day_id
                FROM Workout w
                WHERE w.user_id = ? 
                ORDER BY w.  date DESC
            """, (user_id,))
            workouts = training_cur.fetchall()
            
            print(f"    Найдено {len(workouts)} тренировок")
            
            for workout in workouts:
                training_day_name = get_training_day_name(training_conn, workout['training_day_id'])
                
                context = f"""Тренировка {user_name} - {training_day_name}
Дата: {workout['date']}
Длительность: {workout['duration']} минут
Общий объем: {workout['total_volume']} кг
Упражнения: 
"""
                # Получаем упражнения для этой тренировки
                training_cur.execute("""
                    SELECT 
                        e.id,
                        e. name,
                        e.description,
                        e.difficulty,
                        e.muscle_group_primary,
                        e.muscle_group_secondary,
                        ee.sets,
                        ee.reps,
                        ee.weight,
                        ee.rest_time_sec,
                        ee.note
                    FROM ExercisesEntry ee
                    JOIN Exercise e ON ee.exercise_id = e.id
                    WHERE ee.training_day_id = ?  
                """, (workout['training_day_id'],))
                
                exercises = training_cur.fetchall()
                
                if exercises:
                    for ex in exercises:
                        weight_str = f"{ex['weight']} кг" if ex['weight'] else "без веса"
                        muscle_groups = ex['muscle_group_primary']
                        if ex['muscle_group_secondary']: 
                            muscle_groups += f", {ex['muscle_group_secondary']}"
                        
                        context += f"""
  • {ex['name']}
    Группы мышц: {muscle_groups}
    Описание: {ex['description'] if ex['description'] else 'нет'}
    Сложность: {ex['difficulty']}/10
    Выполнение: {ex['sets']} сеты x {ex['reps']} повторений
    Вес: {weight_str}
    Отдых: {ex['rest_time_sec']} сек
    Примечание:  {ex['note'] if ex['note'] else 'нет'}
"""
                else:
                    context += "  Нет упражнений\n"
                
                # Сохраняем контекст в RAG БД
                try:
                    rag_cur.execute(f"""
                        INSERT INTO {CONTEXT_TABLE} 
                        (user_id, context_text, source_table, source_id)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, context, 'Workout', workout['id']))
                    context_count += 1
                except Exception as e:
                    print(f"    ✗ Ошибка при сохранении контекста тренировки: {e}")
            
            # 2. Контекст из PROGRESS (все записи)
            training_cur.execute("""
                SELECT 
                    p.id,
                    p.date,
                    p. weight,
                    p.total_volume
                FROM Progress p
                WHERE p.user_id = ?
                ORDER BY p. date DESC
            """, (user_id,))
            progress_records = training_cur.fetchall()
            
            if progress_records:
                progress_context = f"Прогресс {user_name}:\n"
                for prog in progress_records:
                    progress_context += f"  {prog['date']}:  Вес тела {prog['weight']} кг | Общий объем {prog['total_volume']} кг\n"
                
                try:
                    rag_cur.execute(f"""
                        INSERT INTO {CONTEXT_TABLE} 
                        (user_id, context_text, source_table, source_id)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, progress_context, 'Progress', 0))
                    context_count += 1
                    print(f"    ✓ Прогресс сохранен ({len(progress_records)} записей)")
                except Exception as e: 
                    print(f"    ✗ Ошибка при сохранении контекста прогресса: {e}")
            
            # 3. Контекст из доступных упражнений
            training_cur.execute("""
                SELECT 
                    id,
                    name,
                    description,
                    difficulty,
                    muscle_group_primary,
                    muscle_group_secondary,
                    type,
                    equipment
                FROM Exercise
                ORDER BY muscle_group_primary
            """)
            exercises = training_cur. fetchall()
            
            if exercises:
                exercise_context = f"Справочник упражнений для {user_name}:\n\n"
                
                current_muscle = None
                for ex in exercises: 
                    muscle = ex['muscle_group_primary']
                    
                    if muscle != current_muscle:
                        exercise_context += f"\n{muscle. upper()}:\n"
                        current_muscle = muscle
                    
                    secondary = f" (+ {ex['muscle_group_secondary']})" if ex['muscle_group_secondary'] else ""
                    equipment = f" [{ex['equipment']}]" if ex['equipment'] else ""
                    exercise_context += f"  • {ex['name']}{secondary} ({ex['type']}){equipment} - сложность {ex['difficulty']}/10\n"
                    
                    if ex['description']:
                        exercise_context += f"    {ex['description'][: 100]}.. .\n"
                
                try:
                    rag_cur.execute(f"""
                        INSERT INTO {CONTEXT_TABLE} 
                        (user_id, context_text, source_table, source_id)
                        VALUES (?, ?, ?, ?)
                    """, (user_id, exercise_context, 'Exercise', 0))
                    context_count += 1
                    print(f"    ✓ Справочник упражнений сохранен ({len(exercises)} упражнений)")
                except Exception as e:
                    print(f"    ✗ Ошибка при сохранении контекста упражнений: {e}")
        
        rag_conn.commit()
        print(f"\n✓ Контекст успешно создан:  {context_count} записей\n")
        
    except Exception as e:
        print(f"✗ Ошибка при генерации контекста: {e}")
        import traceback
        traceback.print_exc()
        rag_conn.rollback()
        return 0
    
    finally:
        training_conn.close()
        rag_conn.close()
    
    return context_count
