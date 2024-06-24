# Importa le librerie necessarie
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Definisce una pipeline per un classificatore Random Forest
random_forest = Pipeline([
    # Il primo passo è un vettorizzatore CountVectorizer che trasforma il testo in una matrice di conteggi di termini
    ('vect', CountVectorizer()),
    # Il secondo passo è il classificatore RandomForestClassifier
    ('clf', RandomForestClassifier()),
])
