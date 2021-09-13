from reading_datasets import *
from AAOSVM import *

# dataframes = [ds1_sub, ds2, ds3, ds4]
# dataframes = [ds2, ds3, ds4]
# dataframes = [ds3, ds4]
dataframes = []

X_train, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train, y)

y[y == 0] = -1

# Add an outlier
X_outlier = np.append(X_train_scaled, [0.1, 0.1])
X_outlier = X_outlier.reshape(X_train.shape[0] + 1, X_train.shape[1])
y_outlier = np.append(y, 1)

# Add a second outlier, so the first one is showed
X_outlier = np.append(X_outlier, [0.1, 0.1])
X_outlier = X_outlier.reshape(X_train.shape[0] + 2, X_train.shape[1])
y_outlier = np.append(y_outlier, 0)

# print(X_outlier)
# print(y_outlier)

datasets = []
for ds in dataframes:
    # TODO: reserve 200 phishing for generating adversarial samples
    y = ds.pop(ds.columns.tolist()[-1])
    y = y.to_numpy()
    x = ds.to_numpy()
    datasets.append((x, y))


def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN) / len(y)


# Adding mock datasets
datasets.append((X_train_scaled, y))
datasets.append((X_outlier, y_outlier))

np.random.seed(0)

# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    print("\n\nGoing through DS" + str(ds_cnt + 1))

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

    # # Initialize error cache
    # initial_error = model.decision_function(model.X) - model.y
    # model.errors = initial_error

    # model.train()
    ################################################################
    # Slidding window
    Sx = ds[0][0]
    Sy = ds[1][0]
    Sx = np.append([Sx], [ds[0][1]], 0)
    Sy = np.append([Sy], [ds[1][1]], 0)

    # Initialize model
    model = AAOSVM(Sx, Sy, C, initial_alphas, initial_b, np.zeros(m))
    model.w = initial_w

    # X = ds[0]
    # Y = ds[1]
    for i in range(2, len(ds[1]), 1):
        x = ds[0][i]
        y = ds[1][i]

        # if i == len(ds[1]) - 1 or i == len(ds[1]) - 2:
        #     print(x, y)
        # model.predi
        Sx, Sy = model.partial_fit(Sx, Sy, x, y, i)

        print("reached final {}".format(i))

    # # Support vector count
    # sv_count = np.where(model.alphas > 0)[0]
    # # sv_count = np.where(model.alphas > 0)
    # print(sv_count)
    # input()
    # # Make prediction
    # y_hat = model.predict(x)

    # # Calculate accuracy
    # acc = calc_acc(y, y_hat)

    print("Support vector count: %d" % (np.count_nonzero(model.alphas)))
    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    # print("accuracy:\t%.3f" % (acc))
    # print("Converged after %d iterations" % (iterations))
    # print("Time to train:\t\t" + str(toc - tic))
    print(model.alphas.sum())

    fig, ax = plt.subplots()
    grid, ax = model.plot_decision_boundary(ax)

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
