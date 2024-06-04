from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

decision_tree = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', DecisionTreeClassifier()),
])
