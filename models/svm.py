# Importa le librerie necessarie
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

# Definisce una pipeline per un classificatore SVC (Support Vector Classifier)
svc = Pipeline([
    # Il primo passo è un vettorizzatore CountVectorizer che trasforma il testo in una matrice di conteggi di termini
    ('vect', CountVectorizer()),
    # Il secondo passo è il classificatore SVC
    ('clf', SVC()),
])
