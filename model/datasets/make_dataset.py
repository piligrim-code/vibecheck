import pandas as pd
import sys
import os

def process_chat(input_file, output_file, chat_name, num_of_lines):
    """
    Описание:
    - Читает CSV-файл, оставляет строки с msg_type == 'text'.
    - Оставляет только указанные колонки.
    - Добавляет колонку chat_name.
    - Сохраняет последние num_of_lines строк.
    - Конкатенирует с существующим output файлом (если есть) и сохраняет результат.

    Параметры:
    - input_file: путь к входному файлу CSV.
    - output_file: путь к выходному файлу CSV.
    - chat_name: название чата, которое будет добавлено в колонку 'chat_name'.
    - num_of_lines: количество последних строк, которые будут сохранены.
    """
    # Читаем входной файл
    df = pd.read_csv(input_file)
    
    # Оставляем строки, где msg_type == 'text', и только нужные колонки
    df_text = df.loc[df['msg_type'] == 'text'][['sender', 'sender_id', 'date', 'msg_type', 'msg_content']]
    
    # Добавляем колонку chat_name со значением chat_name
    df_text['chat_name'] = chat_name
    
    # Оставляем последние num_of_lines строк
    df_text = df_text.tail(num_of_lines)
    
    # Проверяем, существует ли output файл и не является ли он пустым
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            # Читаем существующий output файл, если он не пустой
            df_output = pd.read_csv(output_file)
            # Конкатенируем данные
            df_text = pd.concat([df_output, df_text], ignore_index=True)
        except pd.errors.EmptyDataError:
            # Если файл пустой, просто записываем новый df_text
            pass
    
    # Сохраняем результат в output файл
    df_text.to_csv(output_file, index=False)

if __name__ == '__main__':
    # Подсказка для пользователя
    if len(sys.argv) != 5:
        print("Использование: python process_chat.py <input_file> <output_file> <chat_name> <num_of_lines>")
        print("Пример: python process_chat.py input.csv output.csv 'My Chat' 100")
        sys.exit(1)
    
    # Аргументы командной строки
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    chat_name = sys.argv[3]
    num_of_lines = int(sys.argv[4])
    
    # Запуск функции
    process_chat(input_file, output_file, chat_name, num_of_lines)
