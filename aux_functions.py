from sklearn.metrics import accuracy_score, f1_score, recall_score

# from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# import matplotlib.pyplot as plt
from random import randrange
from csv import writer
import numpy as np
import pickle
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

# from numpy.lib.function_base import average

# from ray.util.joblib import register_ray
# from joblib import parallel_backend
# from ray.services import get_node_ip_address
# from ray import init

# get_node_ip_address = lambda: '127.0.0.1'


def dataset_split(X: np.ndarray, y: np.ndarray, split: int, type, random_state=42):
    """
    Split the dataset in training and testing according to
    a number of samples, or a percentage of dataset to be reserved for testing

    Parameters:
        X (np.ndarray): dataset to be split into train and test.
        y (np.array): labels of the dataset to be split into train and test.
        split (int): number of samples to be used as test.
        type 
            (int): class to remove random 'split' samples (phishing or legitimate).
            (list/np.array): list/np.array to select 'split' samples.
        random_state (int): initial random state for auditatorial purposes.

        Returns:
            X_train (np.ndarray): split of the dataset to train.
            X_test (np.ndarray): split of the dataset to test.
            y_train (np.array): split of the labels of the dataset to train.
            y_test (np.array): split of the labels of the dataset to test.
            selected (list/np.array): indexes of the test split.
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

        X_test, y_test = X[selected], y[selected]
        X_train = np.delete(X, selected, 0)
        y_train = np.delete(y, selected)

        # selected = np.random.choice(len(y[y == type]), split)

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X[y == type], y[y == type], test_size=split, random_state=random_state
        # )

    # Specific instances from a class, with possible random instances to fill in
    # assumes all the instances are from a single class!
    # it is not checking if list is made from a single class
    elif isinstance(type, list) or isinstance(type, np.ndarray):
        if split > len(type):
            # split -= len(type)
            t = y[type[0]]

            selected = set(type)

            rng = np.random.default_rng(random_state)
            sz_y = len(y)
            sz_s = 0
            while sz_s < split:
                aux = rng.integers(low=0, high=sz_y)
                if y[aux] == t:
                    selected.add(aux)
                    sz_s = len(selected)

            selected = list(selected)

            X_test, y_test = X[selected], y[selected]
            X_train = np.delete(X, selected, 0)
            y_train = np.delete(y, selected)
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X[y == y[type[0]]], y[y == y[type[0]]], test_size=split, random_state=random_state
            # )

        elif len(type) == split:
            X_test, y_test = X[type], y[type]
            X_train = np.delete(X, type, 0)
            y_train = np.delete(y, type)
            selected = type

        elif split < len(type):
            # selected = np.random.choice(type, split, replace=False)
            rng = np.random.default_rng(random_state)
            selected = rng.choice(type, split, replace=False)

            X_test, y_test = X[selected], y[selected]
            X_train = np.delete(X, selected, 0)
            y_train = np.delete(y, selected)

    return X_train, X_test, y_train, y_test, selected


# def dataset_splits(X: np.ndarray, y: np.ndarray, split, type, random_state=42):
#     """

#     """
#     # saving the results on a csv file
#     f = open("standard_splits.csv", "w", newline="")
#     wrt = writer(f)
#     header = ["Split", "Indexes"]
#     wrt.writerow(header)

#     for i in range(10):
#         X_train, X_test, y_train, y_test, selected = dataset_split(X, y, split, type, i)
#         wrt.writerow([i] + selected)


def cross_validate(clf, X, y, type, clf_name, cv=5, random_state=42, aux='', reset=None):
    """
    Cross validation to be used across all classifiers
    TODO: continue explanation

    Parameters:
        clf (classifier): classifier to be trained and evaluated.
        X (np.ndarray): dataset to be used in the cross validation.
        y (np.array): labels of dataset to be used in the cross validation.
        type (str): classifier type; decides how the cross validation is going to happen.
        clf_name (str): classifier name.
        cv (int): number of cross validations to be performed.
        random_state (int): initial random state for auditatorial purposes.
        aux (str): extra parameter to help specify the filename.
        reset (dict): dictionary of initial parameters of the classifier to reset it.

    Returns:
        ACCs (np.array): list of accuracy scores from the cross validation.
        TPRs (np.array): list of TPR scores from the cross validation.
        F1s (np.array): list of f1 scores from the cross validation.
        loss (np.array): list of numbers of samples that did not get predicted correctly (label ~(-1|+1)).
    """
    ACCs = np.zeros(cv)
    TPRs = np.zeros(cv)
    F1s = np.zeros(cv)
    loss = np.zeros(cv)
    j = 0

    kf = KFold(n_splits=cv, random_state=42, shuffle=True)

    f = open(clf_name + aux + "_{}_{}.pkl".format(type, random_state), "wb")

    if type == "inc":
        for train_index, test_index in kf.split(X, y):
            X_partial, X_hold = X[train_index], X[test_index]
            y_partial, y_hold = y[train_index], y[test_index]

            clf.reset()

            # TODO: change to go 1 sample at a time
            clf.fit(X_partial, y_partial, classes=[-1, 1])
            y_pred = clf.predict(X_hold)
            neg = y_pred == -1
            pos = y_pred == 1
            bin = neg | pos

            pickle.dump(clf, f)

            print(len(y_pred[neg]) + len(y_pred[pos]))
            print(len(y_pred[~bin]))
            print("y_pred", j)

            ACCs[j] = accuracy_score(y_hold[bin], y_pred[bin])
            TPRs[j] = recall_score(y_hold[bin], y_pred[bin])
            F1s[j] = f1_score(y_hold[bin], y_pred[bin])
            loss[j] = len(y_pred[~bin])

            j += 1

    elif type == "std":
        # register_ray()
        for train_index, test_index in kf.split(X, y):
            X_partial, X_hold = X[train_index], X[test_index]
            y_partial, y_hold = y[train_index], y[test_index]

            clf.set_params(**reset)

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
            print("y_pred", j)

            ACCs[j] = accuracy_score(y_hold[bin], y_pred[bin])
            TPRs[j] = recall_score(y_hold[bin], y_pred[bin])
            F1s[j] = f1_score(y_hold[bin], y_pred[bin])
            loss[j] = len(y_pred[~bin])

            j += 1

    elif type == "AAOSVM":
        for train_index, test_index in kf.split(X, y):

            # Saving scores
            h = open("AAOSVM_scores" + aux + "_{}.csv".format(random_state), "w", newline="")
            wrt_s = writer(h)
            header = [
                "Utility of Malicious sample",
                "Utility of Regular sample",
                "Cost of Malicious sample",
                "Cost of Regular sample",
            ]
            wrt_s.writerow(header)
            # /Saving scores

            # splitting the data
            X_partial, X_hold = X[train_index], X[test_index]
            y_partial, y_hold = y[train_index], y[test_index]

            # resetting the classifier
            clf = clf.reset(X_partial, y_partial)

            # creating a slidding window
            Sx = clf.X
            Sy = clf.y
            sz = len(y_partial)
            for i in range(2, sz, 1):
                x = X_partial[i]
                Y = y_partial[i]

                # training on one sample at a time
                Sx, Sy = clf.partial_fit(X_partial, Sx, Sy, x, Y, i)

                # Saving scores
                wrt_s.writerow(
                    [clf.Em, clf.Er, clf.Ym, clf.Yr,]
                )

                # showing progress at the rate of 1%
                if i % (sz // 100) == 0:
                    print("reached final {}".format(i))

            y_pred = clf.decision_function(X_hold)
            y_pred[y_pred < 0] = -1
            y_pred[y_pred >= 0] = 1
            neg = y_pred == -1
            pos = y_pred == 1
            bin = neg | pos

            # saving trained model
            pickle.dump(clf, f)

            print(len(y_pred[neg]) + len(y_pred[pos]))
            print(len(y_pred[~bin]))
            print("y_pred", j)

            ACCs[j] = accuracy_score(y_hold[bin], y_pred[bin])
            TPRs[j] = recall_score(y_hold[bin], y_pred[bin])
            F1s[j] = f1_score(y_hold[bin], y_pred[bin])
            loss[j] = len(y_pred[~bin])

            j += 1

    elif type == "OSVM":
        for train_index, test_index in kf.split(X, y):
            X_partial, X_hold = X[train_index], X[test_index]
            y_partial, y_hold = y[train_index], y[test_index]

            w = 2

            clf.set_params(**reset)

            # Slidding window
            Sx = X_partial[0]
            Sy = y_partial[0]
            Sx = np.append([Sx], [X_partial[1]], 0)
            Sy = np.append([Sy], [y_partial[1]], 0)

            sz = len(y_partial)
            for i in range(2, sz, 1):
                x = X_partial[i]
                Y = y_partial[i]

                clf = clf.partial_fit(Sx)

                if i % (sz // 100) == 0:
                    # if i % 50 == 0:
                    print("reached final {}".format(i))

                if w < 100:
                    w += 1

                # if reached the limit of the window
                # i.e. window is full and now will move
                if Sy.shape[0] >= w:
                    Sx = Sx[1:]
                    Sy = Sy[1:]

                # Adding instance to the window
                Sx = np.append(Sx, [x], 0)
                Sy = np.append(Sy, [Y], 0)

            y_pred = clf.predict(X_hold)
            # y_pred[y_pred < 0] = -1
            # y_pred[y_pred >= 0] = 1
            neg = y_pred == -1
            pos = y_pred == 1
            bin = neg | pos

            pickle.dump(clf, f)

            print(len(y_pred[neg]) + len(y_pred[pos]))
            print(len(y_pred[~bin]))
            print("y_pred", j)

            ACCs[j] = accuracy_score(y_hold[bin], y_pred[bin])
            TPRs[j] = recall_score(y_hold[bin], y_pred[bin])
            F1s[j] = f1_score(y_hold[bin], y_pred[bin])
            loss[j] = len(y_pred[~bin])

            j += 1

    f.close()

    return ACCs, TPRs, F1s, loss  # last one means the "ones that should not be"


def feature_selection(x, f=None):
    """
    Gets a list of features to feed into the AST.

    Parameters:
        x (np.array): a sample.
        f (int): a number of features to be selected from the sample.

    Returns:
        sel_features (list): list of features selected.
    """
    if f == None:
        f = randrange(0, 5)

    sel_features = []

    while len(sel_features) < f:
        s = randrange(0, x.shape[0])
        if s not in sel_features:
            sel_features.append(s)

    return sel_features


# # Feature selection testing
# selFeatures = feature_selection(np.array(list(range(30))), 2)
# print(selFeatures)
# selFeatures = feature_selection(np.ones(30), 2)
# print(selFeatures)
# # /Feature selection testing

# # Bias metrics
def class_imbalance(y):
    """
    Measures the class imbalance using the formula

                (p - n)
                -------
                (p + n)
    
        where
        p == number of positive instances
        n == number of negative instances

    Parameters:
        y (np.array): array of labels.
    
    Return:
        (float): result of the formula.
    """

    p = sum(y[y == 1])
    n = -sum(y[y == -1])

    return (p - n) / (p + n)


def DPPTL(attribute: int, a, X, y):
    """
    Difference in Positive Proportions of True Labels
    also checks Demographic Parity when == 0
    
    uses the following formula:
    pa/na - pd/nd

        where
        pa == number of positive instances where an attribute has a certain value
        na == number of instances where an attribute has a certain value
        pd == number of positive instances where an attribute does not have a certain value
        nd == number of instances where an attribute does not have a certain value

    Parameters:
        attribute (int): an attribute to be looked at.
        a (float): value of the attribute to be looked at.
        X (np.ndarray): dataset.
        y (np.array): predicted labels of dataset.

    Return:
        (float): result of the formula.
    """
    # positive instances
    p = X[y == 1]

    # indexes where the attribute has 'a' as value
    index_pa = p[:, attribute] == a
    index_na = X[:, attribute] == a

    return p[index_pa] / X[index_na] - p[~index_pa] / X[~index_na]


# def demographic_parity(attribute, a, X, y):
#     '''
#     attribute -> an attribute to be looked at
#     a -> value of the attribute to be looked at
#     X -> dataset
#     y -> predicted labels
#     '''
#     pass


def equality_of_opportunity(attribute, a, X, y):
    """
    attribute -> an attribute to be looked at
    a -> value of the attribute to be looked at
    X -> dataset
    y -> predicted labels
    """
    pass


def empirical_robustness(
    classifier, x: np.array, adv_x: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Compute the Empirical Robustness of a classifier object over the sample 'x' for a given adversarial crafting
    method 'attack'. This is equivalent to computing the minimal perturbation that the attacker must introduce for a
    successful attack.
    | Paper link: https://arxiv.org/abs/1511.04599
    | Adapted from https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/metrics/metrics.py

    Parameters:
        classifier (classifier): A trained model.
        x (np.array): Data sample of shape that can be fed into 'classifier' and was used to generate the adversarial samples.
        adv_x (np.ndarray): A set of samples adversarialy generated from 'x'.

    Returns:
        (Union[float, np.ndarray]): The average empirical robustness computed on 'x'.
    """
    # crafter = get_crafter(classifier, attack_name, attack_params)
    # crafter.set_params(**{"minimal": True})
    # adv_x = crafter.generate(x)

    # Predict the labels for adversarial examples
    y = classifier.predict(x)
    y_pred = classifier.predict(adv_x)

    idxs = np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)
    if np.sum(idxs) == 0.0:
        return 0.0

    norm_type = 2
    # if hasattr(crafter, "norm"):
    #     norm_type = crafter.norm  # type: ignore
    perts_norm = np.linalg.norm(
        (adv_x - x).reshape(x.shape[0], -1), ord=norm_type, axis=1
    )
    perts_norm = perts_norm[idxs]

    return np.mean(
        perts_norm
        / np.linalg.norm(x[idxs].reshape(np.sum(idxs), -1), ord=norm_type, axis=1)
    )


# # /Bias metrics

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
