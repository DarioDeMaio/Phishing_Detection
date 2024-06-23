import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from models.random_forest import random_forest
from models.decision_tree import decision_tree
from models.mnb import mnb
from models.knn import knn
from models.svm import svc
from models.bert import bert_classifier
from time import time

def main():
    df = pd.read_csv('data/Phishing_Email.csv')
    df = df.dropna()

    X = df['email'].to_numpy()
    y = df['label'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    begin = time()
    print('Random Forest')
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Time taken:', time() - begin)

    begin = time()
    print('Multinomial Naive Bayes')
    mnb.fit(X_train, y_train)
    y_pred = mnb.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Time taken:', time() - begin)

    begin = time()
    print('K-Nearest Neighbors')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Time taken:', time() - begin)

    begin = time()
    print('SVC')
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Time taken:', time() - begin)

    begin = time()
    print('Decision Tree')
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Time taken:', time() - begin)

    begin = time()
    print('Bert')
    y_pred = bert_classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Time taken:', time() - begin)

if __name__ == '__main__':
    main()
