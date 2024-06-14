from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

svc = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', SVC()),
])
