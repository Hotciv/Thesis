from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from random import randrange
from csv import writer
import numpy as np
import pickle

# from numpy.lib.function_base import average

# from ray.util.joblib import register_ray
# from joblib import parallel_backend
# from ray.services import get_node_ip_address
# from ray import init

# get_node_ip_address = lambda: '127.0.0.1'

def dataset_split(X: np.ndarray, y: np.ndarray, split, type, random_state=42):
    """
    Split the dataset in training and testing according to
    a number of samples, or a percentage of dataset to be reserved for testing
    """

    selected = None

    # Random instances from a class
    if isinstance(type, int):
        selected = set()
        rng = np.random.default_rng(random_state)
        sz_y = len(y)
        sz_s = 0
        while sz_s < split:
            aux = rng.integers(low=0, high=sz_y)
            if y[aux] == type:
                selected.add(aux)
                sz_s = len(selected)

        selected = list(selected)
        
        X_test, y_test = \
            X[selected], y[selected]
        X_train = np.delete(X, selected, 0)
        y_train = np.delete(y, selected)

        # selected = np.random.choice(len(y[y == type]), split)

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X[y == type], y[y == type], test_size=split, random_state=random_state
        # )

    # Specific instances from a class, with possible random instances to fill in
    elif isinstance(type, list):
        # probably the only occurences
        if len(type) < split:
            split -= len(type)
            X_train, X_test, y_train, y_test = train_test_split(
                X[y == y[type[0]]], y[y == y[type[0]]], test_size=split, random_state=random_state
            )

        elif len(type) == split:
            X_test, y_test = \
                X[type], y[type]
            X_train = np.delete(X, type, 0)
            y_train = np.delete(y, type)
    return X_train, X_test, y_train, y_test, selected

def dataset_splits(X: np.ndarray, y: np.ndarray, split, type, random_state=42):
    # saving the results on a csv file
    f = open("standard_splits.csv", "w", newline="")
    wrt = writer(f)
    header = ["Split", "Indexes"]
    wrt.writerow(header)

    for i in range(10):
        X_train, X_test, y_train, y_test, selected = dataset_split(X, y, split, type, i) 
        wrt.writerow([i] + selected)


def cross_validate(clf, X, y, type, clf_name, cv=5, random_state=42):
    ACCs = np.zeros(cv)
    TPRs = np.zeros(cv)
    F1s = np.zeros(cv)
    loss = np.zeros(cv)
    j = 0

    kf = KFold(n_splits=cv, random_state=42, shuffle=True)
    # kf = KFold(n_splits=cv)

    f = open(clf_name + "_{}_{}.pkl".format(type,random_state), 'wb')

    if type == 'inc':
        for train_index, test_index in kf.split(X, y):
            X_partial, X_hold = X[train_index], X[test_index]
            y_partial, y_hold = y[train_index], y[test_index]

            clf.fit(X_partial, y_partial, classes=[-1,1])
            y_pred = clf.predict(X_hold)
            neg = y_pred == -1
            pos = y_pred == 1
            bin = neg | pos

            pickle.dump(clf, f)

            print(len(y_pred[neg]) + len(y_pred[pos]))
            print(len(y_pred[~bin]))
            print('y_pred', j)

            ACCs[j] = accuracy_score(y_hold[bin], y_pred[bin])
            TPRs[j] = recall_score(y_hold[bin], y_pred[bin])
            F1s[j] = f1_score(y_hold[bin], y_pred[bin])
            loss[j] = len(y_pred[~bin])

            j += 1
            
    elif type == 'std':
        # register_ray()
        for train_index, test_index in kf.split(X, y):
            X_partial, X_hold = X[train_index], X[test_index]
            y_partial, y_hold = y[train_index], y[test_index]

            # init(address='auto')
            # with parallel_backend('ray'):
            clf.fit(X_partial, y_partial)
            y_pred = clf.predict(X_hold)
            neg = y_pred == -1
            pos = y_pred == 1
            bin = neg | pos

            pickle.dump(clf, f)

            print(len(y_pred[neg]) + len(y_pred[pos]))
            print(len(y_pred[~bin]))
            print('y_pred', j)

            ACCs[j] = accuracy_score(y_hold[bin], y_pred[bin])
            TPRs[j] = recall_score(y_hold[bin], y_pred[bin])
            F1s[j] = f1_score(y_hold[bin], y_pred[bin])
            loss[j] = len(y_pred[~bin])

            j += 1

    elif type == 'AAOSVM':
        for train_index, test_index in kf.split(X, y):
            # print(train_index)
            # print(test_index)
            # print(y)
            # input()

            X_partial, X_hold = X[train_index], X[test_index]
            y_partial, y_hold = y[train_index], y[test_index]

            clf = clf.reset(X_partial, y_partial)

            Sx = clf.X
            Sy = clf.y
            sz = len(y_partial)
            for i in range(2, sz, 1):
                x = X_partial[i]
                Y = y_partial[i]

                Sx, Sy = clf.partial_fit(X_partial, Sx, Sy, x, Y, i)

                if i % (sz // 100) == 0:
                # if i % 50 == 0:
                    print("reached final {}".format(i))

            y_pred = clf.decision_function(X_hold)
            y_pred[y_pred < 0] = -1
            y_pred[y_pred >= 0] = 1
            neg = y_pred == -1
            pos = y_pred == 1
            bin = neg | pos

            pickle.dump(clf, f)

            print(len(y_pred[neg]) + len(y_pred[pos]))
            print(len(y_pred[~bin]))
            print('y_pred', j)

            ACCs[j] = accuracy_score(y_hold[bin], y_pred[bin])
            TPRs[j] = recall_score(y_hold[bin], y_pred[bin])
            F1s[j] = f1_score(y_hold[bin], y_pred[bin])
            loss[j] = len(y_pred[~bin])

            j += 1

    f.close()

    return ACCs, TPRs, F1s, loss  # last one means the "ones that should not be"


def feature_selection(x, f=None):
    if f == None:
        f = randrange(0, 5)

    selFeatures = []

    while len(selFeatures) < f:
        s = randrange(0, x.shape[0])
        if s not in selFeatures:
            selFeatures.append(s)

    return selFeatures

# # Feature selection testing
# selFeatures = feature_selection(np.array(list(range(30))), 2)
# print(selFeatures)
# selFeatures = feature_selection(np.ones(30), 2)
# print(selFeatures)
# # /Feature selection testing

# def plot_columns(X, y=None):
#     sz = len(data)
#     values = np.unique(data[0])
#     X = np.arange(len(values))
#     fig = plt.figure()
#     ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#     # plt.axis([min(values), max(values), 0, 5000])

#     colors = ['b', 'r', 'g']

#     for i in range(sz):
#         values, counts = np.unique(data[i], return_counts=True)
#         ax.bar(X + 0.40*i, counts, color = colors[i], width = 0.8/sz)

#     plt.show()