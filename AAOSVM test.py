from aux_functions import *

from reading_datasets import *
from AAOSVM import *

from csv import writer
from time import time
import pickle

# dataframes = [ds1_sub, ds2, ds3, ds4]
# dataframes = [ds2, ds3, ds4]
# dataframes = [ds4]
# dataframes = []

# # Mock dataset
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
# y_outlier = np.append(y_outlier, -1)

# # print(X_outlier)
# # print(y_outlier)
# # /Mock dataset

# datasets = []
# for ds in dataframes:
#     y = ds.pop(ds.columns.tolist()[-1])
#     y = y.to_numpy()
#     x = ds.to_numpy()
#     datasets.append((x, y))

# # Adding mock datasets
# datasets.append((X_train_scaled, y))
# datasets.append((X_outlier, y_outlier))
# # /Adding mock datasets

# ds4 normalized
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y = ds1_sub.pop(ds1_sub.columns.tolist()[-1])
y = y.to_numpy()
ds5 = scaler.fit_transform(ds1_sub)

datasets = [(ds5, y)]
# /ds4 normalized

np.random.seed(0)

# Saving the results
f = open("AAOSVM, ds1_norm 10x random200.csv", "w", newline="")
wrt = writer(f)
header = ["Dataset", "# of critical instances", "ACC", "TPR", "F1", "Time to execute"]
wrt.writerow(header)

g = open('indexes - ds1, norm.pkl', 'wb')
# /Saving the results

# TODO: average results and get standard deviations from files
# repeat experiment 10x
for k in range(10):

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):

        X, y = ds

        X_train, X_test, y_train, _, _ = dataset_split(X, y, 200, 1, k)
        # X_train, X_test, y_train, _, _ = dataset_split(X, y, 200, 1)

        print("\n\nGoing through DS" + str(ds_cnt + 1) + ' ' + str(k) + ' times')

        # Set model parameters and initial values
        C = 100.0
        # C = 1000.0  # for SVM_w_SMO comparison
        # C = 1.0  # for SVM_w_SMO comparison
        # m = len(ds[1])  # for SVM_w_SMO comparison
        m = 2
        initial_alphas = np.zeros(m)
        initial_b = 0.0
        initial_w = np.zeros(len(ds[0][0]))
        initial_Psi = 0.0

        # # Initialize model
        # model = AAOSVM(ds[0], ds[1], C,
        #             initial_alphas, initial_b, np.zeros(m))
        # model.w = initial_w

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
        
        model.w = initial_w

        # X = ds[0]
        # Y = ds[1]
        start_time = time()
        ACCs, TPRs, F1s, _ = cross_validate(model, ds[0], ds[1], 'AAOSVM', 'AAOSVM', 
                        # random_state=int(format(11, 'b') + format(ds_cnt, 'b'), 2))
                        random_state=int(format(k, 'b') + format(ds_cnt, 'b'), 2))
        finish = time() - start_time

        print('finished', finish)
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
        header = ["Dataset", "# of critical instances", "ACC", "TPR", "F1", "Time to execute"]
        wrt.writerow([ds_cnt, np.count_nonzero(model.alphas), ACCs.mean(), TPRs.mean(), F1s.mean(), finish])
        pickle.dump(np.where(model.alphas > 0)[0], g)
        print('wrote')

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
