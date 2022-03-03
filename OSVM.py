from sklearn.linear_model import SGDOneClassSVM

from sklearn.preprocessing import StandardScaler

# from adversarial_samples import cost, generating_adversarial_samples
from reading_datasets import *
from aux_functions import *
from sklearn import metrics
from csv import writer
from time import time
import numpy as np

# dataframes = [ds1_sub, ds2, ds3, ds4]
# dataframes = [ds2, ds3, ds4]
# dataframes = [ds1_sub]
# dataframes = []

datasets = to_dataset()

np.random.seed(0)

# Saving the results
f = open("OSVM, ds1-4 10x random200.csv", "w", newline="")
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
# /Saving the results

# # TODO: average results and get standard deviations from files
# # repeat experiment 10x
for k in range(10):
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        model = SGDOneClassSVM(random_state=42)  # TODO: remove random_state

        X, y = ds

        # X_train, X_test, y_train, _, _ = dataset_split(X, y, 200, 1, k)
        X_train, X_test, y_train, _, _ = dataset_split(X, y, 200, 1)

        # print("\n\nGoing through DS" + str(ds_cnt + 1) + ' ' + str(k) + ' times')
        print("\n\nGoing through DS" + str(ds_cnt + 1) + " times")

        start_time = time()
        ACCs, TPRs, F1s, _ = cross_validate(
            model,
            X_train,
            y_train,
            "OSVM",
            "inc",
            # random_state=int(format(11, 'b') + format(ds_cnt, 'b'), 2))
            random_state=int(format(k, "b") + format(ds_cnt, "b"), 2),
        )
        finish = time() - start_time

        print("finished", finish)

        # for i in range(2, len(ds[1]), 1):
        #     x = ds[0][i]
        #     y = ds[1][i]

        #     model.partial_fit(Sx, Sy)
        # score.append(abs(model.predict(np.reshape(x, (1, -1))) + y))

        # header = ["Dataset", "ACC", "ACC std", "TPR", "TPR std", "F1", "F1 std", "Time to execute"]
        # print(x + ' ' for x in header)
        wrt.writerow(
            [
                ds_cnt,
                ACCs.mean(),
                ACCs.std(),
                TPRs.mean(),
                TPRs.std(),
                F1s.mean(),
                F1s.std(),
                finish,
            ]
        )
        # wrt = [ds_cnt, ACCs.mean(), TPRs.mean(), F1s.mean(), finish]
        # print(x + ' ' for x in wrt)

f.close()
