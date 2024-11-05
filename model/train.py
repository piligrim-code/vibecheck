import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import clearml
from clearml import Task, Logger
def calculate_rmse(outputs, scores, criterion):
    rmse = torch.sqrt(criterion(outputs.squeeze(), scores.to(device).float())).mean()
    return rmse.item()

def train_step(model, optimizer, criterion, dataloader, epoch, device):
    model.train()
    train_loss = 0.0
    train_rmse = 0.0
    train_len = len(dataloader)
    for tweets, scores in tqdm(dataloader, total=train_len, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
        inputs = tokenizer_deep.batch_encode_plus(
            tweets,
            return_tensors='pt',
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        # Получение предсказаний
        outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

        # Вычисление потерь и обратное распространение
        loss = criterion(outputs.squeeze(), scores.to(device).float()).to(device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        train_rmse += calculate_rmse(outputs, scores, criterion)

    train_loss_avg = train_loss / train_len
    train_rmse_avg = train_rmse / train_len
    return train_loss_avg, train_rmse_avg

def validation_step(model, criterion, dataloader, epoch, device):
    model.eval()
    val_loss = 0
    val_rmse = 0
    val_len = len(dataloader)

    with torch.no_grad():
        for tweets, scores in tqdm(dataloader, total=val_len, desc=f"Validation", leave=False):
            inputs = tokenizer_deep.batch_encode_plus(
                tweets,
                return_tensors='pt',
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            # Получение предсказаний
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

            # Вычисление потерь
            val_loss += criterion(outputs.squeeze(), scores.to(device).float()).item()
            val_rmse += calculate_rmse(outputs, scores, criterion)

    val_loss_avg = val_loss / val_len
    val_rmse_avg = val_rmse / val_len
    return val_loss_avg, val_rmse_avg

task = Task.init(project_name='Bertv2', task_name='BertTrain')
logger = Logger.current_logger()
best_val_loss = float('inf')

# Цикл обучения
for epoch in range(epochs):
    train_loss, train_rmse = train_step(bert_regression_model, optimizer, criterion, train_dataloader, epoch, device)
    val_loss, val_rmse = validation_step(bert_regression_model, criterion, val_dataloader, epoch, device)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Train RMSE: {train_rmse:.3f}, Val Loss: {val_loss:.3f}, Val RMSE: {val_rmse:.3f}")

    # Логирование метрик в ClearML
    logger.report_scalar(title='Train Loss', series='loss', value=train_loss, iteration=epoch)
    logger.report_scalar(title='Train RMSE', series='rmse', value=train_rmse, iteration=epoch)
    logger.report_scalar(title='Validation Loss', series='loss', value=val_loss, iteration=epoch)
    logger.report_scalar(title='Validation RMSE', series='rmse', value=val_rmse, iteration=epoch)

    # Сохранение модели
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(bert_regression_model.state_dict(), 'bertweight.pth')
    if epoch == epochs - 1:  # Проверяем,  последняя ли это эпоха
        torch.save(bert_regression_model.state_dict(), 'bertweight.pth')
        task.upload_artifact('model', 'bertweight.pth')

# Завершение задачи ClearML
task.close()
