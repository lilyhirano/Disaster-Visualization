import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


lbp_mean = np.load("npy_files/lbp_mean_2.npy")
entropy_mean = np.load("npy_files/entropy_mean_2.npy")
blue = np.load("npy_files/mean_B_2.npy")
green = np.load("npy_files/mean_G_2.npy")

features = ['lbp_mean', 'entropy_mean', 'blue', 'green']
X = np.concatenate([ lbp_mean,  entropy_mean, blue, green], axis=1)
Y = np.load("npy_files/2_MERGED_labels.npy")

label_legend = {
    0: "midwest-flooding",
    1: "socal-fire"
}

def stratified_CV(X, Y, model, n_split=5, metric=accuracy_score):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)

    values = []
    for train_index, test_index in skf.split(X, Y):
        print("Running Kfold...")
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        values.append(metric(Y_test, Y_pred, average="macro"))

    return np.mean(values), values, confusion_matrix(Y_test, Y_pred)

def run_model(X, Y, n):
    
    model = GradientBoostingClassifier(n_estimators=500, max_depth=3,min_samples_split=5, 
                                       min_samples_leaf=3, subsample=0.8, max_features='sqrt')
    mean_score, values, matrix = stratified_CV(X, Y, model, n_split= n, metric=f1_score)


    print(f"\nK1 Score = {mean_score}")
    print(f"K1 Values = {values}")


    '''
    Create Confusion Matrix
    '''
    labels = [label_legend[y] for y in np.unique(Y)]
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(cmap="Oranges", values_format='d', xticks_rotation= 45)
    plt.title("Disaster Types: Confusion Matrix for the Final Fold")
    plt.tight_layout()
    plt.savefig("images/dType_confusion_matrix.png", format="png")
    plt.close()

    '''
    Create Per-Fold F1 Scores Figure
    '''
    plt.plot(np.arange(1, n+1), values, marker='o', color= 'darkorange')
    plt.xlabel("Fold")
    plt.ylabel("Weighted F1 Score")
    plt.title("Disaster Types: Per-Fold F1 Scores")
    plt.ylim(0,1)
    plt.xticks(np.arange(1, n+1))
    plt.savefig("images/dType_KFold.png", format="png")
    plt.close()

    '''
    Create Importances Figure
    '''
    importances = model.feature_importances_
    plt.figure(figsize=(10,6))
    plt.bar(range(len(importances)), importances, color="darkorange")
    plt.xticks(range(len(importances)), features)
    plt.ylabel("Feature Importance")
    plt.ylabel("Feature Type")
    plt.title("Disaster Types: Gradient Boosted Tree Feature Importances")
    plt.savefig("images/dType_importances.png", format="png")
    plt.close()
    

run_model(X,Y, 5)



