from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

knn = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', KNeighborsClassifier(n_neighbors=1)),
])
