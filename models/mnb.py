# Importa le librerie necessarie
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

# Definisce una pipeline per un classificatore Multinomial Naive Bayes
mnb = Pipeline([
    # Il primo passo è un vettorizzatore CountVectorizer che trasforma il testo in una matrice di conteggi di termini
    ('vect', CountVectorizer()),
    # Il secondo passo è il classificatore MultinomialNB
    ('clf', MultinomialNB()),
])
