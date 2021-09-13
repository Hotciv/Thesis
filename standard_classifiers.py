# Code source: Gaël Varoquaux
#              Andreas Müller
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Modified for documentation by Jaques Grobler
# Modified for use by Victor Barboza
# License: BSD 3 clause

# from pyspark.sql import SparkSession # instantiate spark session
from skdist.distribute.search import DistGridSearchCV

# from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from adversarial_samples import cost, generating_adversarial_samples
# from sklearn.utils import shuffle
from reading_datasets import *
from aux_functions import *
from sklearn import metrics
from csv import writer
from time import time
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# # For DistGridSearchCV to work
# spark = (
#     SparkSession
#     .builder
#     .getOrCreate()
#     )
# sc = spark.sparkContext

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
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
    GradientBoostingClassifier(),
]

# Loading datasets
# X - list of list of features
# y - list of classes
# dataframes = [ds1_sub, ds2, ds3, ds4]
dataframes = [ds2, ds3, ds4]
# dataframes = [ds1_sub]
datasets = []

for ds in dataframes:
    y = ds.pop(ds.columns.tolist()[-1])
    # y = y.values.tolist()
    y = y.to_numpy()
    # X = ds.values.tolist()
    X = ds.to_numpy()
    datasets.append((X, y))

# saving the results on a csv file
f = open("standard_classifiers_results.csv", "w", newline="")
wrt = writer(f)
# header = ["Dataset", "Classifier", "ACC", "TPR", "f1", "Time to execute"]
header = ["Dataset", "Classifier", "ACC", "Time to execute"]
wrt.writerow(header)

# TODO: average results and get standard deviations from files
# repeat experiment 10x
for k in range(10):

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        # TODO: preprocess dataset, split into training and test part?
        X, y = ds

        # no scalling for now
        # X = StandardScaler().fit_transform(X)

        # dataset_split(X, y, [])
        X_train, X_test, y_train, _ = dataset_split(X, y, 200)
        # dataset_split(X, y, 0.1)
        # print(len(y_train), len(y_test))
        # input()
        # X_train, y_train= (X, y)

        print("\n\nGoing through DS" + str(ds_cnt + 1) + " " + str(k + 1) + " time")

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            foolers = []
            y = 1  # 1 or -1
            start_time = time()
            
            score = cross_validate(clf, X_train, y_train)
            # ACCs = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
            finish = time() - start_time

            selFeatures = feature_selection(X_test[0], 4)
            # print(selFeatures)
            # input()
            X = pd.DataFrame(X)
            
            for instance in X_test:
                samples = generating_adversarial_samples(instance, selFeatures, X, y)
                pred = clf.predict(samples)
                if ((pred + y) == 0).any():
                    foolers.append(pred)

            print(len(foolers)/200)
            # print(score, ACCs)
            # input()
            # clf.fit(X_train, y_train)
            # score = clf.score(X_test, y_test)
            # print(str(start_time - time()) + " score " + str(score))

            # f1s = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'f1')
            # TPRs = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'recall')

            # param_grid = {}
            # ACCs = DistGridSearchCV(
            #             clf, param_grid,
            #             sc=sc, scoring='accuracy',
            #             verbose=True
            #             )

            # header = ['Dataset', 'Classifier', 'ACC', 'TPR', 'f1', 'Time to execute']
            # wrt.writerow([ds_cnt, name, ACCs.mean(), TPRs.mean(), f1s.mean(), time() - start_time])
            wrt.writerow([ds_cnt, name, score.mean(), finish])

f.close()
