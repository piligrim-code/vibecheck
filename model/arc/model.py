import torch
import torch.nn as nn
from transformers import AutomModel

class BertRegression(nn.Module):
    def __init__(self, model_name="DeepPavlov/rubert-base-cased",
                 pretrained_weights_path="destination_filename.pth", device = 'cpu'):

        super(BertRegression, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)  # Предобученная модель BERT
        self.dropout = torch.nn.Dropout(0.2)  # Регуляризация
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)  # Добавление Layer Normalization
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.load_state_dict(
            torch.load(pretrained_weights_path,
                       map_location=torch.device('cpu'),
                       weights_only=True))
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropout_output = self.dropout(pooled_output)
        layer_norm_output = self.layer_norm(dropout_output)
        linear_output = self.regressor(layer_norm_output)
        return linear_output 
