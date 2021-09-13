from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from random import randrange
import numpy as np

def dataset_split(X:np.ndarray, y:np.ndarray, split, random_state=42):
    """
    Split the dataset in training and testing according to
    a number of samples, or a percentage of dataset to be reserved for testing
    """

    if isinstance(split, list):
        # Split into training, validation and testing
        pass
    elif isinstance(split, float):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=random_state
        )
    elif isinstance(split, int):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=split, random_state=random_state
        )
    return X_train, X_test, y_train, y_test


def cross_validate(clf, X, y, cv=5, scoring="accuracy", random_state=42):
    scores = np.zeros(cv)
    j = 0

    # kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)
    kf = KFold(n_splits=cv)

    for train_index, test_index in kf.split(X, y):
        X_partial, X_hold = X[train_index], X[test_index]
        y_partial, y_hold = y[train_index], y[test_index]

        for i in range(cv):
            if scoring == "accuracy":
                clf.fit(X_partial, y_partial)
                scores[j] = clf.score(X_hold, y_hold)
        
        j +=1

    return scores


def feature_selection(x, f=None):
    if f == None:
        f = randrange(0, 5)

    selFeatures = []

    while len(selFeatures) < f:
        s = randrange(0, x.shape[0])
        if s not in selFeatures:
            selFeatures.append(s)

    return selFeatures