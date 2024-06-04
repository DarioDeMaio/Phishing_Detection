from sklearn.base import BaseEstimator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.optim import Adam
from torch.nn import BCELoss
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class BertClassifier(BaseEstimator):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('ealvaradob/bert-finetuned-phishing')
        self.model = AutoModelForSequenceClassification.from_pretrained('ealvaradob/bert-finetuned-phishing', num_labels = 2)
        
        
    def fit(self, X, y):
        train_encodings = self.tokenizer(list(X), truncation = True, padding = True, max_length = 512)
        train_labels = torch.tensor(y)
        train_inputs = torch.tensor(train_encodings['input_ids'])
        train_dataset = TensorDataset(train_encodings, train_labels)
        batch_size = 16
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        optimizer = Adam(self.model.parameters(), lr = 2e-05)
        criterion = BCELoss()
        self.model.train()
        epochs = 4
        
        for i in range(epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()
                inputs = batch[0]
                labels = batch[2]
                
                outputs = self.model(inputs)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                
                del inputs, labels
                torch.cuda.empty_cache()
        
    def predict(self, X):
        predictions = []
        test_encodings = self.tokenizer(list(X), truncation = True, padding = True, max_length = 512)
        # print(test_encodings)
        test_inputs = torch.tensor(test_encodings['input_ids'])
        test_dataset = TensorDataset(test_inputs)
        test_dataloader = DataLoader(test_dataset)
        
        self.model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                input = batch[0]
                outputs = self.model(input)
                # outputs = 1 if outputs > 0.5 else 0
                predictions.append(outputs)
                del input, outputs
                torch.cuda.empty_cache()
                
        return np.array(predictions)
    
bert_classifier = BertClassifier()