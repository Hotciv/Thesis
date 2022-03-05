# Classifiers
from art.defences.detector.evasion import BinaryActivationDetector, BinaryInputDetector
from art.estimators.classification.scikitlearn import ScikitlearnSVC

# Datasets
from reading_datasets import *

# Save results
from csv import writer

# Measure time
from time import time

# Miscellaneous
from aux_functions import *

# Loading datasets
# X - list of list of features
# y - list of classes
# dataframes = [ds1_sub, ds2, ds3, ds4]
dataframes = [ds2, ds3, ds4]
# dataframes = [ds1_sub]
datasets = []

for ds in dataframes:
    y = ds.pop(ds.columns.tolist()[-1])
    y = y.to_numpy()
    X = ds.to_numpy()
    datasets.append((X, y))

names = [
    'Binary Activation'
    "Binary Input"
]

classifiers = [
    BinaryActivationDetector(ScikitlearnSVC),
    BinaryInputDetector()
]

# saving the results on a csv file
f = open("incremental_classifiers_results.csv", "w", newline="")
wrt = writer(f)
header = ["Dataset", "Classifier", "ACC", "TPR", "F1", "Time to execute"]
# header = ["Dataset", "Classifier", "ACC", "Time to execute"]
wrt.writerow(header)

# TODO: average results and get standard deviations from files
# repeat experiment 10x
for k in range(10):

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        # TODO: preprocess dataset, split into training and test part?
        X, y = ds

        # X_train, X_test, y_train, _ = dataset_split(X, y, 200)
        X_train, y_train = X, y

        print("\n\nGoing through DS" + str(ds_cnt + 1) + " " + str(k + 1) + " time")

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            foolers = []
            y = 1  # 1 or -1
            
            print(name)

            start_time = time()
            ACCs, TPRs, F1s, _ = cross_validate(clf, X_train, y_train, 'std')
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

            # header = ['Dataset', 'Classifier', 'ACC', 'TPR', 'f1', 'Time to execute']
            wrt.writerow([ds_cnt, name, ACCs.mean(), TPRs.mean(), F1s.mean(), finish])
            print('wrote')
            # wrt.writerow([ds_cnt, name, score.mean(), finish])

f.close()
