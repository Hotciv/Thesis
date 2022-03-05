from sklearn.linear_model import SGDOneClassSVM

from reading_datasets import *
from aux_functions import *
from csv import writer
from time import time
import numpy as np

# dataframes = [ds1_sub, ds2, ds3, ds4]
datasets, name = to_dataset()

np.random.seed(0)

# Saving the results
f = open("OSVM, " + name + " 10x random200.csv", "w", newline="")
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

for k in range(10):
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        model = SGDOneClassSVM(random_state=42) 

        reset = model.get_params()

        X, y = ds

        X_train, X_test, y_train, _, _ = dataset_split(X, y, 200, 1, k)

        print("\n\nGoing through DS" + str(ds_cnt + 1) + ' ' + str(k) + ' times')

        start_time = time()
        ACCs, TPRs, F1s, _ = cross_validate(
            model,
            X_train,
            y_train,
            "OSVM",
            "inc",
            random_state = k,
            aux = "_ds{}".format(ds_cnt + 1),
            reset = reset,
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
        print("wrote")
        # wrt = [ds_cnt, ACCs.mean(), TPRs.mean(), F1s.mean(), finish]
        # print(x + ' ' for x in wrt)

f.close()
