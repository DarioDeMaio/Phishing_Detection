# Importa le librerie necessarie
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Definisce una pipeline per un classificatore ad albero di decisione
decision_tree = Pipeline([
    # Il primo passo è un vettorizzatore CountVectorizer che trasforma il testo in una matrice di conteggi di termini
    ('vect', CountVectorizer()),
    # Il secondo passo è il classificatore DecisionTreeClassifier
    ('clf', DecisionTreeClassifier()),
])
