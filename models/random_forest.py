from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

RandomForest = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier()),
])
