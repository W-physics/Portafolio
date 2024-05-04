import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

def label_data():

    data = pd.read_csv('data.csv')

    closes = data['Closes']
    y = np.zeros(len(closes))

    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            y[i] = 1
    
    return data, y

#Standarization and split of the data

def preprocessing():

    pipeline = Pipeline([('std_scaler', StandardScaler())])

    data, y = label_data()
    X = pipeline.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test


def search_best(X_train, y_train):
    
    svm_clf = LinearSVC(dual='auto')

    params = dict(C=np.linspace(0,1,10))

    grid_search = GridSearchCV(svm_clf, params, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_

    return best_estimator


