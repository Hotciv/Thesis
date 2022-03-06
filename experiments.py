##########################################
#############--- imports ---##############
##########################################

from reading_datasets import *
from aux_functions import *
from csv import writer, reader
from time import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import SGDOneClassSVM

# # for helping the experiment 0?
# from pyspark.sql import SparkSession # instantiate spark session
# from skdist.distribute.search import DistGridSearchCV
# # /for helping the experiment 0?

##########################################
##########--- general settings---#########
##########################################

# Loading datasets
# X - list of list of features
# y - list of classes
# dataframes = [ds1_sub, ds2, ds3, ds4]

np.random.seed(0)

datasets, name = to_dataset()

# input -> experiment number
n = int(input("What is the experiment? [0-4]\n"))
while n < 0 and n > 4:
    n = int(input("What is the experiment? [0-4]\n"))

# input -> normalize datasets?
normalization = input("Use normalized dataset? (y/n)\n")
while normalization != "y" and normalization != "n":
    normalization = input("Use normalized dataset? (y/n)\n")

if normalization == "y":
    # iterate over datasets and normalize each one of them
    for ds_cnt, ds in enumerate(datasets):
        X, y = ds
        datasets[ds_cnt] = (dataset_normalization([X])[0], y)

def filename_build(n: int, selection="10x random200"):
    """
    Function to create an automated filename for the experiment
    """
    if normalization == "y":
        norm = ", normalized"
    else:
        norm = ""

    if n == 1:
        fn = "standard_classifiers_results "
    elif n == 2:
        fn = "OSVM, "

    return fn + name + " " + selection + norm + ".csv"

##############################################
################--- TODO: ---#################
# --- experiment 0 - getting the best shot ---#
##############################################

# experiment 0.1 - normalization
# experiment 0.2 - tunning

# # For DistGridSearchCV to work
# spark = (
#     SparkSession
#     .builder
#     .getOrCreate()
#     )
# sc = spark.sparkContext

# # somewhere in the training bits
# param_grid = {}
# ACCs = DistGridSearchCV(
#             clf, param_grid,
#             sc=sc, scoring='accuracy',
#             verbose=True
#             )

# experiment 0.3 - tunning on normalization

##########################################
# --- experiment 1 - standard, batch ---###
##########################################

def experiment_cv(
    n: int, wrt: writer, X, y, k=11, just_SVMs=False, selection="10x random200"
):
    """
    The core of experiments cross validation and saves results in\
    csv files with filenames automatically generated.

    Parameters:
        n (int): experiment number.
        wrt (writer): csv writer to save the results.
        k (int): iteration number; 11 is for when there is just one execution.
        just_SVMs
            (bool): as False, uses all classifiers available and\
                as True, uses just SVMs.
            (list): list of indexes of classifiers to be used.
        selection (["10x random200", "support_vectors"]): selection of\
            random 200 phishing samples or\
            support vectors + close by samples (KNN)
    """
    if selection == "10x random200":
        # removing 200 random phishing samples
        X_train, X_test, y_train, y_test, selected = dataset_split(X, y, 200, 1, k)

    # settings
    if n == 1:
        names = [
            "Nearest Neighbors (KNN)",
            "Linear SVM",
            "RBF SVM",
            "Decision Tree",
            "Random Forest",
            "Naive Bayes",
            "Gradient Boost",
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=100, max_iter=10000, random_state=42),
            SVC(gamma=2, C=100, max_iter=10000, random_state=42),
            DecisionTreeClassifier(max_depth=5, random_state=42),
            RandomForestClassifier(
                max_depth=5, n_estimators=10, max_features=1, random_state=42
            ),
            GaussianNB(),
            GradientBoostingClassifier(random_state=42),
        ]

        clf_type = "std"

        if just_SVMs == True:
            names = names[1:3]
            classifiers = classifiers[1:3]

    elif n == 2:
        names = ["inc"]
        classifiers = [SGDOneClassSVM(random_state=42)]
        clf_type = "OSVM"
    # /settings

    # iterate over classifiers
    for clf_name, clf in zip(names, classifiers):

        print(clf_name)

        # getting initial parameters to reset classifier in the cross validation
        reset = clf.get_params()

        # timed cross validation
        start_time = time()
        ACCs, TPRs, F1s, loss = cross_validate(
            clf,
            X_train,
            y_train,
            clf_type,
            clf_name,
            random_state=k,
            aux="_ds{}".format(ds_cnt + 1),  # TODO: indicate normalization
            reset=reset,
        )
        finish = time() - start_time

        print("finished", finish)

        # saving
        # header  # to peek at definition
        wrt.writerow(
            [
                ds_cnt,
                clf_name,
                ACCs.mean(),
                ACCs.std(),
                TPRs.mean(),
                TPRs.std(),
                F1s.mean(),
                F1s.std(),
                finish,
                loss,
            ]
        )
        print("wrote")


def experiment_load(
    n: int, k=11, all_models=False, just_SVMs=False, selection="10x random200"
):
    """
    Load experiments from cross validation and returns model, the selected 200 samples, 
    and the test/train from the original cross validation split.

    Parameters:
        n (int): experiment number.
        k (int): iteration number; 11 is for when there is just one execution.
        just_SVMs
            (bool): as False, uses all classifiers available and
                as True, uses just SVMs.
            (list): list of indexes of classifiers to be used.
        selection (["10x random200", "support_vectors"]): selection of
            random 200 phishing samples or
            support vectors + close by samples (KNN)

    Yields:
        clf 
            (classifier): trained classifier from a cross validation.
            (list): all trained classifiers from a cross validation.
        the_200 (np.ndarray): the selected 200 samples.
        X_test (np.ndarray): split of the dataset to test.
        y_test (np.array): split of the labels of the dataset to test.
        selected (list/np.array): indexes of the test split.
    """
    from glob import glob

    if selection == "10x random200":
        # removing 200 random phishing samples
        X_train, X_test, y_train, y_test, selected = dataset_split(X, y, 200, 1, k)
        the_200 = X_test

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    # settings
    if n == 1:
        names = [
            "Nearest Neighbors (KNN)",
            "Linear SVM",
            "RBF SVM",
            "Decision Tree",
            "Random Forest",
            "Naive Bayes",
            "Gradient Boost",
        ]

        clf_type = "std"

        if just_SVMs == True:
            names = names[1:3]
            classifiers = classifiers[1:3]

    elif n == 2:
        names = ["inc"]
        clf_type = "OSVM"
    # /settings

    # iterate over classifiers
    for clf_name in names:
        # loading the first model from a pkl file
        f = open(
            clf_name + "_ds{}".format(ds_cnt + 1) + "_{}_{}.pkl".format(clf_type, k),
            "rb",
        )

        if not all_models:
            # loads first model from a cross validation
            clf = pickle.load(f)

            # gets the test/train from the original cross validation split
            train_index, test_index = kf.split(X_train, y_train).__next__()
            X_partial, X_hold = X_train[train_index], X_train[test_index]
            y_partial, y_hold = y_train[train_index], y_train[test_index]
        else:
            clf = []
            ACCs = np.zeros(kf.get_n_splits())
            TPRs = np.zeros(kf.get_n_splits())
            F1s = np.zeros(kf.get_n_splits())
            loss = np.zeros(kf.get_n_splits())
            j = 0
            for train_index, test_index in kf.split(X_train, y_train):
                _, X_hold = X_train[train_index], X_train[test_index]
                _, y_hold = y_train[train_index], y_train[test_index]

                # loads all models from a cross validation
                clf.append(pickle.load(f))

                y_pred = clf[j].predict(X_hold)
                # y_pred[y_pred < 0] = -1
                # y_pred[y_pred >= 0] = 1
                neg = y_pred == -1
                pos = y_pred == 1
                bin = neg | pos

                ACCs[j] = accuracy_score(y_hold[bin], y_pred[bin])
                TPRs[j] = recall_score(y_hold[bin], y_pred[bin])
                F1s[j] = f1_score(y_hold[bin], y_pred[bin])
                loss[j] = len(y_pred[~bin])

                j += 1

            print(
                [
                    ds_cnt,
                    clf_name,
                    ACCs.mean(),
                    ACCs.std(),
                    TPRs.mean(),
                    TPRs.std(),
                    F1s.mean(),
                    F1s.std(),
                    loss,
                ],
                "\n",
            )  # TODO: reconstruct csv file with proper clf_name

        return clf, the_200, X_hold, X_hold, selected


header = [
    "Dataset",
    "Classifier",
    "ACC",
    "ACC std",
    "TPR",
    "TPR std",
    "F1",
    "F1 std",
    "Time to execute",
    "Loss",
]
# saving the results on a csv file
f = open(filename_build(n), "w", newline="")
wrt = writer(f)
wrt.writerow(header)

# repeating the experiment 10x on different splits
for k in range(10):

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        X, y = ds

        print("\n\nGoing through DS" + str(ds_cnt + 1) + " " + str(k + 1) + " time")

        experiment_cv(n, wrt, X, y, k)
        # loaded = experiment_load(2, k, True)
        # print(loaded)
        # input()

        # feature selection and etc
        # selFeatures = feature_selection(X_test[0], 4)
        # # print(selFeatures)
        # # input()
        # X = pd.DataFrame(X)

        # for instance in X_test:
        #     samples = generating_adversarial_samples(instance, selFeatures, X, y)
        #     pred = clf.predict(samples)
        #     if ((pred + y) == 0).any():
        #         foolers.append(pred)

        # print(len(foolers)/200)

        # print(score, ACCs)
        # input()
        # clf.fit(X_train, y_train)
        # score = clf.score(X_test, y_test)
        # print(str(start_time - time()) + " score " + str(score))
f.close()

# experiment 5
