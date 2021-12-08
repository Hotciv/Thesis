from reading_datasets import *
from SVM_example import *
from time import time

# dataframes = [ds1_sub, ds2, ds3, ds4]
dataframes = [ds2, ds3, ds4]
# dataframes = [ds1_sub]
datasets = []
for ds in dataframes:
    # TODO: reserve 200 phishing for generating adversarial samples
    y = ds.pop(ds.columns.tolist()[-1])
    # X = ds
    y = y.to_numpy()
    X = ds.to_numpy()
    datasets.append((X, y))

# repeat experiment 10x
for k in range(10):
    for ds_cnt, ds in enumerate(datasets):
            X, y = ds
            print("\n\nGoing through DS" + str(ds_cnt + 1) + ' ' + str(k + 1) + ' time')

            start_time = time()
            w, b = train_svr(X, y)
            print('b = %.4f\t|time = %.4f' % b, time() - start_time)
