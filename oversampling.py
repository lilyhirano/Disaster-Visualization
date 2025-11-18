import numpy as np
'''
Oversamples a certain disaster label by copying and flipping it
'''

def oversample_label(X,Y ,d_label = 2):

    indexes = np.where(Y == 2)

    X_over = X[indexes]

    X_flip = np.flip(X_over, axis = 2)

    y_flip = np.full(len(X_flip), d_label)

    X_out = np.concatenate([X, X_flip], axis=0)
    Y_out = np.concatenate([Y, y_flip], axis=0)

    return X_out, Y_out 