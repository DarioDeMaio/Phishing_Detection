# Importa le librerie necessarie
from sklearn.base import BaseEstimator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Definisce una classe per il classificatore BERT che eredita da BaseEstimator di scikit-learn
class BertClassifier(BaseEstimator):

    # Inizializza il classificatore
    def __init__(self):
        # Carica il tokenizer e il modello pre-addestrato di BERT
        self.tokenizer = AutoTokenizer.from_pretrained('ealvaradob/bert-finetuned-phishing')
        self.model = AutoModelForSequenceClassification.from_pretrained('ealvaradob/bert-finetuned-phishing', num_labels=2)
        # Imposta il dispositivo su GPU se disponibile, altrimenti su CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Congela i parametri del modello base di BERT
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        # Sblocca i parametri del classificatore finale
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    # Funzione per addestrare il modello
    def fit(self, X, y):
        # Tokenizza i dati di addestramento
        train_encodings = self.tokenizer(list(X), truncation=True, padding=True, max_length=512, return_tensors='pt')
        train_labels = torch.tensor(y).to(self.device)
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
        # Definisce il batch size e il DataLoader per l'addestramento
        batch_size = 16
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # Definisce l'ottimizzatore e la funzione di perdita
        optimizer = Adam(self.model.parameters(), lr=2e-5)
        criterion = CrossEntropyLoss()
        self.model.train()
        # Imposta il numero di epoche di addestramento
        epochs = 4

        # Loop di addestramento
        for epoch in range(epochs):
            for batch in train_dataloader:
                optimizer.zero_grad()
                inputs = batch[0].to(self.device)
                masks = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                # Ottiene le previsioni del modello e calcola la perdita
                outputs = self.model(input_ids=inputs, attention_mask=masks)
                loss = criterion(outputs.logits, labels)
                # Esegue il backpropagation e l'aggiornamento dei pesi
                loss.backward()
                optimizer.step()

                # Libera la memoria della GPU
                del inputs, masks, labels
                torch.cuda.empty_cache()

    # Funzione per fare previsioni con il modello addestrato
    def predict(self, X):
        self.model.eval()
        predictions = []
        # Tokenizza i dati di test
        test_encodings = self.tokenizer(list(X), truncation=True, padding=True, max_length=512, return_tensors='pt')
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'])
        test_dataloader = DataLoader(test_dataset, batch_size=16)

        # Disattiva il calcolo del gradiente per il test
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch[0].to(self.device)
                masks = batch[1].to(self.device)

                # Ottiene le previsioni del modello
                outputs = self.model(input_ids=inputs, attention_mask=masks)
                logits = outputs.logits
                # Calcola le previsioni finali prendendo l'argmax delle logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)

                # Libera la memoria della GPU
                del inputs, masks, logits
                torch.cuda.empty_cache()

        return np.array(predictions)

# Istanzia il classificatore BERT
bert_classifier = BertClassifier()
