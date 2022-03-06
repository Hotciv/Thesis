from sklearn import datasets
from aux_functions import *

from reading_datasets import *
from AAOSVM import *

from csv import writer
from time import time
import pickle

# dataframes = [ds1_sub, ds2, ds3, ds4]
# dataframes = [0, 1, 2, 3]

datasets, name = to_dataset([2])

np.random.seed(0)

# Saving the results
f = open("AAOSVM, " + name + " 10x random200.csv", "w", newline="")
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
    "Number of critical instances",
    "Loss",
]
wrt.writerow(header)

g = open("indexes - " + name + ".pkl", "wb")
# /Saving the results

for k in range(10):

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        X, y = ds

        X_train, X_test, y_train, _, _ = dataset_split(X, y, 200, 1, k)

        print("\n\nGoing through DS" + str(ds_cnt + 1) + " " + str(k) + " times")

        # Set model parameters and initial values
        C = 100.0
        m = 2
        initial_alphas = np.zeros(m)
        initial_b = 0.0
        initial_w = np.zeros(len(ds[0][0]))

        # Slidding window
        Sx = X_train[0]
        Sy = y_train[0]
        Sx = np.append([Sx], [X_train[1]], 0)
        Sy = np.append([Sy], [y_train[1]], 0)

        # Initialize model
        model = AAOSVM(Sx, Sy, C, initial_alphas, initial_b, np.zeros(m))

        # Initialize error cache
        initial_error = model.decision_function(model.X) - model.y
        model.errors = initial_error

        # Initialize weights
        model.w = initial_w

        # X = ds[0]
        # Y = ds[1]
        start_time = time()
        ACCs, TPRs, F1s, loss = cross_validate(
            model,
            X_train,
            y_train,
            "AAOSVM",
            "AAOSVM",
            random_state=k,
            aux="_ds{}".format(ds_cnt + 1),
        )
        finish = time() - start_time

        print("finished", finish)
        # # Support vector count
        # sv_count = np.where(model.alphas > 0)[0]
        # # sv_count = np.where(model.alphas > 0)
        # print(sv_count)
        # input()
        # # Make prediction
        # y_hat = model.predict(x)

        print("Support vector count: %d" % (np.count_nonzero(model.alphas)))
        print("bias:\t\t%.3f" % (model.b))
        print("w:\t\t" + str(model.w))
        # print("accuracy:\t%.3f" % (acc))
        # print("Converged after %d iterations" % (iterations))
        # print("Time to train:\t\t" + str(toc - tic))
        # print(model.alphas.sum())

        # fig, ax = plt.subplots()
        # grid, ax = model.plot_decision_boundary(ax)
        # plt.plot(model.decision_function(ds[0]))
        # plt.show()

        # header = ["Dataset", "# of critical instances", "Time to execute"]
        # wrt.writerow([ds_cnt, np.count_nonzero(model.alphas), finish])

        # header  # to peek at definition
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
                np.count_nonzero(model.alphas),
                loss,
            ]
        )
        pickle.dump((k, ds_cnt, np.where(model.alphas > 0)[0]), g)

        print("wrote")

f.close()
g.close()

################################################################
################# Test on mock dataset #########################
################################################################
# X_train, y = make_blobs(n_samples=1000, centers=2,
#                         n_features=2, random_state=1)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train, y)

# y[y == 0] = -1

# # Set model parameters and initial values
# C = 1000.0
# m = len(X_train_scaled)
# initial_alphas = np.zeros(m)
# initial_b = 0.0

# # Instantiate model
# model = SMOModel(X_train_scaled, y, C,
#                  initial_alphas, initial_b, np.zeros(m))

# # Initialize error cache
# initial_error = model.decision_function(model.X) - model.y
# model.errors = initial_error

# np.random.seed(0)
# model.train()

# fig, ax = plt.subplots()
# grid, ax = model.plot_decision_boundary(ax)

# print(model.alphas.sum())
################################################################
################# Test on mock dataset #########################
################################################################
