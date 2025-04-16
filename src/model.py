import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class NewsClassifier(nn.Module):
    def __init__(self, num_classes, model_name="google/bert_uncased_L-4_H-512_A-8", dropout=0.3):
        super(NewsClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Take [CLS] token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits