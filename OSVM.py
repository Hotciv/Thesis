from sklearn.linear_model import SGDOneClassSVM

from sklearn.datasets import make_blobs, make_circles

from sklearn.preprocessing import StandardScaler
from adversarial_samples import cost, generating_adversarial_samples
from reading_datasets import *
from aux_functions import *
from sklearn import metrics
from csv import writer
from time import time
import numpy as np

# Loading datasets
# X - list of list of features
# y - list of classes
dataframes = [ds1_sub, ds2, ds3, ds4]
# dataframes = [ds2, ds3, ds4]
# dataframes = [ds1_sub]
# dataframes = []
datasets = []

# X_train, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train, y)

# y[y == 0] = -1

# # Add an outlier
# X_outlier = np.append(X_train_scaled, [0.1, 0.1])
# X_outlier = X_outlier.reshape(X_train.shape[0] + 1, X_train.shape[1])
# y_outlier = np.append(y, 1)

# # Add a second outlier, so the first one is showed
# X_outlier = np.append(X_outlier, [0.1, 0.1])
# X_outlier = X_outlier.reshape(X_train.shape[0] + 2, X_train.shape[1])
# y_outlier = np.append(y_outlier, 0)

# # Adding mock datasets
# datasets.append((X_train_scaled, y))
# datasets.append((X_outlier, y_outlier))

for ds in dataframes:
    y = ds.pop(ds.columns.tolist()[-1])
    y = y.to_numpy()
    x = ds.to_numpy()
    datasets.append((x, y))

np.random.seed(0)

# Saving the results
f = open("OSVM, ds1-4 10x random200.csv", "w", newline="")
wrt = writer(f)
header = ["Dataset", "ACC", "TPR", "F1", "Time to execute"]
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
        print("\n\nGoing through DS" + str(ds_cnt + 1) + ' times')

        start_time = time()
        ACCs, TPRs, F1s, _ = cross_validate(model, X_train, y_train, 'OSVM', 'inc', 
                        # random_state=int(format(11, 'b') + format(ds_cnt, 'b'), 2))
                        random_state=int(format(k, 'b') + format(ds_cnt, 'b'), 2))
        finish = time() - start_time

        print('finished', finish)

        # for i in range(2, len(ds[1]), 1):
        #     x = ds[0][i]
        #     y = ds[1][i]

        #     model.partial_fit(Sx, Sy)
            # score.append(abs(model.predict(np.reshape(x, (1, -1))) + y))

        # header = ["Dataset", "ACC", "TPR", "F1", "Time to execute"]
        # print(x + ' ' for x in header)
        wrt.writerow([ds_cnt, ACCs.mean(), TPRs.mean(), F1s.mean(), finish])
        # wrt = [ds_cnt, ACCs.mean(), TPRs.mean(), F1s.mean(), finish]
        # print(x + ' ' for x in wrt)

f.close()

# l = len(score)
# for i in range(1, l, 1):
#     score[i] = score[i] + score[i-1]
#     score[i] = score[i]/(2*l)
# plt.plot(score)
# plt.show()
