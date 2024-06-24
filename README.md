# Email Phishing Detection

## 1. Obiettivo

L'obiettivo principale di questa ricerca è condurre un'analisi comparativa tra diversi approcci per l'identificazione delle email di phishing, al fine di determinare quale modello sia il più efficace, identificare i limiti dei modelli considerati e analizzare i relativi vantaggi e svantaggi. In particolare, si mira a valutare le prestazioni degli algoritmi di machine learning tradizionale e dei modelli di Elaborazione del Linguaggio Naturale (NLP), confrontandoli con i più recenti modelli LLM come Llama.

## 2. Modelli

- **Random Forest**
- **Decision Tree**
- **Multinomial Naive Bayes**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **BERT**
- **LLAMA 7B e 13B**

## 3. Struttura dell'analisi

L'analisi è stata strutturata testando singolarmente ciascun modello elencato sopra. Inoltre, LLAMA 13B è stato testato in combinazione con Random Forest per valutare se la combinazione di un modello LLM prompt based con un modello di machine learning tradizionale possa migliorare le prestazioni nel compito di discriminazione delle email di phishing.

## 4. Struttura della Repository

- **data/**: Contiene i dati utilizzati per l'analisi.
- **models/**: Contiene i modelli addestrati.
- **LM_Studio.ipynb**: Notebook Jupyter per i test e l'analisi dei modelli Llama.
- **__main__.py**: Script principale per l'esecuzione del progetto.
- **requirements.txt**: Elenco delle dipendenze necessarie per eseguire il progetto.

## 5. Requirements

Per eseguire questo progetto, è necessario installare le dipendenze elencate nel file `requirements.txt`. Per installarle, eseguire il seguente comando:

```sh
pip install -r requirements.txt
