import torch
import torch.nn as nn
from transformers import AutoTokenizer
from model.model import BertRegression
tokenizer = AutoTokenizer.from_pretrained("./model/tokenizer_1")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bert_regression_model = BertRegression()


def predict(text, model, tokenizer, device):
  """
  Функция для предсказания модели BERT для заданного текста.

  Args:
      text: Текст для предсказания.
      model: Модель BERT (BertRegression).
      tokenizer: Токенизатор для BERT.
      device: Устройство для вычислений (CPU или GPU).

  Returns:
      Предсказанное непрерывное значение.
  """

  # Токенизация текста
  inputs = tokenizer_deep(text, return_tensors="pt", add_special_tokens=True, padding=True, max_length=512, truncation=True).to(device)
  torch.manual_seed(42)
  # Вывод модели
  with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )

  # Получение предсказания
  output = outputs.squeeze().item()  # Извлекаем выход из последнего слоя (полносвязный)
  return output