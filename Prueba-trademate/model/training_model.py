import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

#Return the ML model in a scikit-learn object

def training_model(X_train, y_train):
    
    svm_clf = LinearSVC(dual='auto')

    params = dict(C=np.arange(1,10))

    grid_search = GridSearchCV(svm_clf, params, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_estimator = grid_search.best_estimator_

    best_estimator.fit(X_train, y_train)

    return best_estimator

# Put 1 (buy) if the price is higher, 0 (sell) other way

def label_data():

    data = pd.read_csv('/home/cod3_breaker/portafolio/Prueba-trademate/data.csv')

    closes = data['Closes']
    y = np.zeros(len(closes))

    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            y[i] = 1
    
    return data, y

#Standarization and split of the data

def preprocessing(data):

    pipeline = Pipeline([('std_scaler', StandardScaler())])
    
    try:
        X = pipeline.fit_transform(data)
    except ValueError:
        X = pipeline.fit_transform(np.reshape(data, (1,-1)))

    return X

def splitting():

    data, y = label_data()
    
    X = preprocessing(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test

#Model performance

def best_score():

    X_train, X_test, y_train, y_test = splitting()

    model = training_model(X_train, y_train)

    print(model.score(X_test, y_test))
