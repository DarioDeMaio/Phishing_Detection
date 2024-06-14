from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

mnb = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])
