from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

random_forest = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', RandomForestClassifier()),
])
