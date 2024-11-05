import re
import pymorphy3

# Инициализация морфологического анализатора для лемматизации
morph = pymorphy3.MorphAnalyzer()

def preprocess_for_annotation(text):
    """
    Предобработка текста для аннотации: приведение к нижнему регистру, 
    удаление ссылок, упоминаний, эмодзи и лишних пробелов.

    Args:
        text (str): Исходный текст.

    Returns:
        str: Очищенный текст.
    
    Example:
        >>> preprocess_for_annotation("Привет! 😊 Это тестовое сообщение с @username и ссылкой https://example.com. 🚀")
        'привет это тестовое сообщение с и ссылкой'
    """
    # Приведение текста к нижнему регистру
    text = text.lower()
    
    # Удаление ссылок
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Удаление упоминаний (например, @username)
    text = re.sub(r'@\w+', '', text)
    
    # Удаление эмодзи
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Эмоции и смайлики
                               u"\U0001F300-\U0001F5FF"  # Символы и пиктограммы
                               u"\U0001F680-\U0001F6FF"  # Транспорт и карты
                               u"\U0001F1E0-\U0001F1FF"  # Флаги (iOS)
                               u"\U00002700-\U000027BF"  # Разные символы
                               u"\U000024C2-\U0001F251"  # Символы внутри текста
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_for_bert(text):
    """
    Предобработка текста для взаимодействия с BERT: приведение к нижнему регистру, 
    удаление ссылок, упоминаний, эмодзи, лишних символов и пробелов, а также лемматизация.

    Args:
        text (str): Исходный текст.

    Returns:
        str: Очищенный и лемматизированный текст.
    
    Example:
        >>> preprocess_for_bert("Привет! 😊 Это тестовое сообщение для @username и ссылка https://example.com. 🚀 Работаем с BERT.")
        'привет это тестовый сообщение для ссылка работать с berto'
    """
    # Приведение текста к нижнему регистру
    text = text.lower()

    # Удаление ссылок
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Удаление упоминаний (например, @username)
    text = re.sub(r'@\w+', '', text)

    # Удаление эмодзи
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Эмоции и смайлики
                               u"\U0001F300-\U0001F5FF"  # Символы и пиктограммы
                               u"\U0001F680-\U0001F6FF"  # Транспорт и карты
                               u"\U0001F1E0-\U0001F1FF"  # Флаги (iOS)
                               u"\U00002700-\U000027BF"  # Разные символы
                               u"\U000024C2-\U0001F251"  # Символы внутри текста
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    # Удаление лишних символов: оставляем только кириллические и латинские буквы, пробелы
    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z\s]', '', text)

    # Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()

    # Лемматизация текста (приведение слов к нормальной форме)
    text = ' '.join([morph.parse(word)[0].normal_form for word in text.split()])

    return text

# Примеры использования
if __name__ == "__main__":
    example_text_annotation = "Привет, 1 ну сейчас бабки работают с дедками! 😊 Это тестовое сообщение с @username и ссылкой https://example.com. 🚀"
    print("Очищенный текст для аннотации:")
    print(preprocess_for_annotation(example_text_annotation))

    example_text_bert = "Привет, 12 ну сейчас бабки работают с дедками! 😊 Это тестовое сообщение для @username и ссылка https://example.com. 🚀 Работаем с BERT."
    print("\nОчищенный текст для BERT:")
    print(preprocess_for_bert(example_text_bert))
