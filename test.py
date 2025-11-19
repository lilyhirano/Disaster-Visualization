import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

from load import run

disaster_list = ["hurricane-matthew", "midwest-flooding", "socal-fire"]
data = run(disaster_list)


X = np.array([img for event in data.values() for img in event["images"]])
Y = np.array([label for event in data.values() for label in event["labels"]])


def oversample_label(X,Y, d_label = 2):

    indexes = np.where(Y == 2)
    X_over = X[indexes]

    X_flip = np.flip(X_over, axis = 2)
    y_flip = np.full(len(X_flip), d_label)

    X_out = np.concatenate([X, X_flip], axis=0)
    Y_out = np.concatenate([Y, y_flip], axis=0)

    return X_out, Y_out 


def prepare_images(X):
    #flattens images
    return X.reshape(X.shape[0], -1) / 255.0


def stratified_CV(X, Y, model, n_split=5, metric=accuracy_score):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

    values = []
    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        values.append(metric(Y_test, Y_pred))

    return np.mean(values), values

def run_model(X, Y):
    
    X = prepare_images(X)
    model = model = LogisticRegression(C=1.0,max_iter=1000,class_weight='balanced',solver='saga')
    mean_score = stratified_CV(X, Y, model)
    print(f"Mean Score = {mean_score:.4f}")


run_model(X,Y)


