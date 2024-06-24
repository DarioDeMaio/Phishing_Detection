# Importa le librerie necessarie
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

# Definisce una pipeline per un classificatore k-Nearest Neighbors
knn = Pipeline([
    # Il primo passo è un vettorizzatore CountVectorizer che trasforma il testo in una matrice di conteggi di termini
    ('vect', CountVectorizer()),
    # Il secondo passo è il classificatore KNeighborsClassifier con k=1 (un vicino)
    ('clf', KNeighborsClassifier(n_neighbors=1)),
])
