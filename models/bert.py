from sklearn.base import BaseEstimator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class BertClassifier(BaseEstimator):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('ealvaradob/bert-finetuned-phishing')
        self.model = AutoModelForSequenceClassification.from_pretrained('ealvaradob/bert-finetuned-phishing', num_labels=2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def fit(self, X, y):
        train_encodings = self.tokenizer(list(X), truncation=True, padding=True, max_length=512, return_tensors='pt')
        train_labels = torch.tensor(y).to(self.device)
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
        batch_size = 16
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.model.parameters(), lr=2e-5)
        criterion = CrossEntropyLoss()
        self.model.train()
        epochs = 4

        for epoch in range(epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()
                inputs = batch[0].to(self.device)
                masks = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                outputs = self.model(input_ids=inputs, attention_mask=masks)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                del inputs, masks, labels
                torch.cuda.empty_cache()

    def predict(self, X):
        self.model.eval()
        predictions = []
        test_encodings = self.tokenizer(list(X), truncation=True, padding=True, max_length=512, return_tensors='pt')
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
        test_dataloader = DataLoader(test_dataset, batch_size=16)

        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch[0].to(self.device)
                masks = batch[1].to(self.device)

                outputs = self.model(input_ids=inputs, attention_mask=masks)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)

                del inputs, masks, logits
                torch.cuda.empty_cache()

        return np.array(predictions)

bert_classifier = BertClassifier()