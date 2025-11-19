import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from skimage.filters import gabor

X = np.load("all_images.npy")
Y = np.load("all_labels.npy")

# X = X[:200]
# Y = Y[:200]

def oversample_label(X,Y, d_label = 2):

    indexes = np.where(Y == 2)
    X_over = X[indexes]

    X_flip = np.flip(X_over, axis = 2)
    y_flip = np.full(len(X_flip), d_label)

    X_out = np.concatenate([X, X_flip], axis=0)
    Y_out = np.concatenate([Y, y_flip], axis=0)

    return X_out, Y_out 

def compute_mean_rgb(X):
    return np.mean(X, axis=(1, 2)) 

def mean_gabor(X, frequency=0.6):
    features = []
    for img in X:
        img_gray = np.mean(img, axis=2)
        filtered, i = gabor(img_gray, frequency=frequency)
        features.append(np.mean(filtered))

    return np.array(features).reshape(-1, 1)


def combine_features(X, frequency = 0.6):
    rgb = compute_mean_rgb(X)
    gabor = mean_gabor(X, frequency)
    
    return np.concatenate([rgb, gabor], axis = 1)


def stratified_CV(X, Y, model, n_split=4, metric=accuracy_score):
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
    
    X_wFeatures = combine_features(X)
    model =LogisticRegression(C=1.0,max_iter=5000,class_weight='balanced',solver='saga')
    mean_score, values = stratified_CV(X_wFeatures, Y, model)
    print(f"Mean Score = {mean_score}")
    print(f"Values = {values}")

run_model(X,Y)
