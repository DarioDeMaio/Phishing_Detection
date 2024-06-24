# Importa le librerie necessarie
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.random_forest import random_forest
from models.decision_tree import decision_tree
from models.mnb import mnb
from models.knn import knn
from models.svm import svc
from models.bert import bert_classifier
from time import time

# Definisce la funzione principale
def main():
    # Legge il file CSV contenente i dati delle email di phishing
    df = pd.read_csv('data/Phishing_Email.csv')
    # Rimuove le righe con valori NaN
    df = df.dropna()

    # Estrae la colonna 'email' come input (X) e la colonna 'label' come output (y)
    X = df['email'].to_numpy()
    y = df['label'].to_numpy()

    # Divide i dati in set di addestramento e test (80% addestramento, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Inizia il timer per Random Forest
    begin = time()
    print('Random Forest')
    # Addestra il modello Random Forest sui dati di addestramento
    random_forest.fit(X_train, y_train)
    # Predice le etichette per il set di test
    y_pred = random_forest.predict(X_test)
    # Stampa il rapporto di classificazione per le previsioni
    print(classification_report(y_test, y_pred))
    # Stampa il tempo impiegato per addestrare e testare il modello
    print('Time taken:', time() - begin)

    # Inizia il timer per Multinomial Naive Bayes
    begin = time()
    print('Multinomial Naive Bayes')
    # Addestra il modello Multinomial Naive Bayes sui dati di addestramento
    mnb.fit(X_train, y_train)
    # Predice le etichette per il set di test
    y_pred = mnb.predict(X_test)
    # Stampa il rapporto di classificazione per le previsioni
    print(classification_report(y_test, y_pred))
    # Stampa il tempo impiegato per addestrare e testare il modello
    print('Time taken:', time() - begin)

    # Inizia il timer per K-Nearest Neighbors
    begin = time()
    print('K-Nearest Neighbors')
    # Addestra il modello K-Nearest Neighbors sui dati di addestramento
    knn.fit(X_train, y_train)
    # Predice le etichette per il set di test
    y_pred = knn.predict(X_test)
    # Stampa il rapporto di classificazione per le previsioni
    print(classification_report(y_test, y_pred))
    # Stampa il tempo impiegato per addestrare e testare il modello
    print('Time taken:', time() - begin)

    # Inizia il timer per SVC (Support Vector Classifier)
    begin = time()
    print('SVC')
    # Addestra il modello SVC sui dati di addestramento
    svc.fit(X_train, y_train)
    # Predice le etichette per il set di test
    y_pred = svc.predict(X_test)
    # Stampa il rapporto di classificazione per le previsioni
    print(classification_report(y_test, y_pred))
    # Stampa il tempo impiegato per addestrare e testare il modello
    print('Time taken:', time() - begin)

    # Inizia il timer per Decision Tree
    begin = time()
    print('Decision Tree')
    # Addestra il modello Decision Tree sui dati di addestramento
    decision_tree.fit(X_train, y_train)
    # Predice le etichette per il set di test
    y_pred = decision_tree.predict(X_test)
    # Stampa il rapporto di classificazione per le previsioni
    print(classification_report(y_test, y_pred))
    # Stampa il tempo impiegato per addestrare e testare il modello
    print('Time taken:', time() - begin)

    # Inizia il timer per Bert (modello pre-addestrato di deep learning)
    begin = time()
    print('Bert')
    # Predice le etichette per il set di test utilizzando il modello Bert pre-addestrato
    y_pred = bert_classifier.predict(X_test)
    # Stampa il rapporto di classificazione per le previsioni
    print(classification_report(y_test, y_pred))
    # Stampa il tempo impiegato per testare il modello
    print('Time taken:', time() - begin)

# Controlla se il modulo è eseguito come programma principale e chiama la funzione main
if __name__ == '__main__':
    main()
