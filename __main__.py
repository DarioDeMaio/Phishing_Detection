import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.random_forest import random_forest
from models.decision_tree import decision_tree

df = pd.read_csv('data/Phishing_Email.csv')
df = df.dropna()

# Categorize the y values
df['label'] = df['label'].map({'Safe Email': 0, 'Phishing Email': 1})

X = df['email'].to_numpy()
y = df['label'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Random Forest')
random_forest.fit(X_train, y_train)
y_pred = random_forest.predict(X_test)
print(classification_report(y_test, y_pred))

print('Decision Tree')
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
print(classification_report(y_test, y_pred))
