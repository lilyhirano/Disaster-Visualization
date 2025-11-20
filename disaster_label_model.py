#https://ds100.org/fa25/gradproject-cv/#milestone-4-project-report-first-draft-20---wednesday-november-19-2025

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc



gabor_mean = np.load("npy_files/gabor_mean_all.npy")
sobel_edge = np.load("npy_files/sobel_edges0.4_all.npy")
sobel_mean = np.load("npy_files/sobel_mean_all.npy")
sobel_var = np.load("npy_files/sobel_var_all.npy")
lbp_mean = np.load("npy_files/lbp_mean_all.npy")
entropy_mean = np.load("npy_files/entropy_mean_all.npy")
blue = np.load("npy_files/mean_B.npy")
green = np.load("npy_files/mean_G.npy")

features = ['gabor','sobel_edge','sobel_mean','sobel_var','lbp_mean', 'entropy_mean', 'blue', 'green']
X = np.concatenate([gabor_mean, sobel_edge, sobel_mean, sobel_var, lbp_mean,  entropy_mean, blue , green], axis=1)
Y = np.load("../satellite-image-data/all_labels.npy")


def stratified_CV(X, Y, model, n_split=5, metric=accuracy_score):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

    values = []
    for train_index, test_index in skf.split(X, Y):
        print("Running Kfold...")
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        values.append(metric(Y_test, Y_pred, average="weighted"))

    return np.mean(values), values, confusion_matrix(Y_test, Y_pred)

def run_model(X, Y, n):
    
    model = GradientBoostingClassifier( n_estimators=500, max_depth=4,min_samples_split=5, 
                                       min_samples_leaf=3, subsample=0.8, max_features='sqrt')
    mean_score, values, matrix = stratified_CV(X, Y, model, n_split= n, metric=f1_score)
    print(f"K1 Score = {mean_score}")
    print(f"Values = {values}")

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=np.unique(Y))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Normalized Confusion Matrix for Disaster Labels")
    plt.savefig("dLabel_confusion_matrix.png", format="png")
    plt.close()

    plt.plot(np.arange(1, n+1), values, marker='o')
    plt.xlabel("Fold")
    plt.ylabel("Weighted F1 Score")
    plt.title("Per-Fold F1 Scores")
    plt.ylim(0,1)
    plt.savefig("dLabel_KFold.png", format="png")
    plt.close()


    importances = model.feature_importances_
    plt.figure(figsize=(10,6))
    plt.bar(range(len(importances)), importances)
    plt.xticks(range(len(importances)), features)
    plt.ylabel("Feature Importance")
    plt.title("Gradient Boosted Tree Feature Importances")
    plt.savefig("dLabel_importances.png", format="png")
    plt.close()
    

run_model(X,Y, 5)