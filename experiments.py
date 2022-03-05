##########################################
#############--- imports ---##############
##########################################

from reading_datasets import *
from aux_functions import *
from csv import writer
from time import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

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

##############################################
################--- TODO: ---#################
# --- experiment 0 - getting the best shot ---#
##############################################


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

# experiment 0.1 - normalization
# experiment 0.2 - tunning
# experiment 0.3 - tunning on normalization

##########################################
# --- experiment 1 - standard, batch ---###
##########################################

# settings
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

# saving the results on a csv file
f = open("standard_classifiers_results " + name + " 10x random200.csv", "w", newline="")
wrt = writer(f)
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
wrt.writerow(header)

# repeating the experiment 10x on different splits
for k in range(10):

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        X, y = ds

        # removing 200 random phishing samples
        X_train, X_test, y_train, _, _ = dataset_split(X, y, 200, 1, k)

        print("\n\nGoing through DS" + str(ds_cnt + 1) + " " + str(k + 1) + " time")

        # iterate over classifiers
        for name, clf in zip(names, classifiers):

            print(name)

            # getting initial parameters to reset classifier in the cross validation
            reset = clf.get_params()

            # timed cross validation
            start_time = time()
            ACCs, TPRs, F1s, loss = cross_validate(
                clf,
                X_train,
                y_train,
                "std",
                name,
                random_state=k,
                aux="_ds{}".format(ds_cnt + 1),
                reset=reset,
            )
            finish = time() - start_time

            print("finished", finish)

            # saving
            # header  # to peek at definition
            wrt.writerow(
                [
                    ds_cnt,
                    name,
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

# experiment 2

# experiment 3

# experiment 4

# experiment 5
