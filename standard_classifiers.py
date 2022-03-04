# Code source: Gaël Varoquaux
#              Andreas Müller
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
# Modified for documentation by Jaques Grobler
# Modified for use by Victor Barboza
# License: BSD 3 clause

# from pyspark.sql import SparkSession # instantiate spark session
# from skdist.distribute.search import DistGridSearchCV

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
    SVC(kernel="linear", C=100, max_iter=10000),
    SVC(gamma=2, C=100, max_iter=10000),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
    GradientBoostingClassifier(),
]

# Loading datasets
# X - list of list of features
# y - list of classes
# dataframes = [ds1_sub, ds2, ds3, ds4]

datasets, name = to_dataset()

# # ds4 normalized
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# y = ds1_sub.pop(ds1_sub.columns.tolist()[-1])
# y = y.to_numpy()
# ds5 = scaler.fit_transform(ds1_sub)

# datasets = [(ds5, y)]
# # /ds4 normalized

# saving the results on a csv file
f = open("standard_classifiers_results - 5, random, SVMs.csv", "w", newline="")
wrt = writer(f)
header = [
    "Dataset",
    "ACC",
    "ACC std",
    "TPR",
    "TPR std",
    "F1",
    "F1 std",
    "Time to execute",
]
wrt.writerow(header)

for k in range(10):

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        
        X, y = ds

        X_train, X_test, y_train, _, _ = dataset_split(X, y, 200, 1, k)
        # X_train, y_train = X, y

        print("\n\nGoing through DS" + str(ds_cnt + 1) + " " + str(k + 1) + " time")

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            foolers = []
            y = 1  # 1 or -1

            print(name)
            
            start_time = time()
            # ACCs, TPRs, F1s, _ = cross_validate(clf, X_train, y_train, 'std', 
            #         random_state=k)
            ACCs, TPRs, F1s, _ = cross_validate(clf, X_train, y_train, 'std', name, 
                    random_state=int(format(k, 'b') + format(ds_cnt, 'b'), 2))
            # ACCs = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
            # f1s = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'f1')
            # TPRs = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'recall')
            finish = time() - start_time

            print('finished', finish)
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


            # param_grid = {}
            # ACCs = DistGridSearchCV(
            #             clf, param_grid,
            #             sc=sc, scoring='accuracy',
            #             verbose=True
            #             )

            # header = ["Dataset", "ACC", "ACC std", "TPR", "TPR std", "F1", "F1 std", "Time to execute"]
            wrt.writerow([ds_cnt, ACCs.mean(), ACCs.std(), TPRs.mean(), TPRs.std(), F1s.mean(), F1s.std(), finish])
            print('wrote')
            # wrt.writerow([ds_cnt, name, score.mean(), finish])

f.close()
