from reading_datasets import *
from AAOSVM import *
from time import time

# dataframes = [ds1_sub, ds2, ds3, ds4]
dataframes = [ds2, ds3, ds4]
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
    return float(TP + TN)/len(y)


# # iterate over datasets
# for ds_cnt, ds in enumerate(datasets):
#     print("\n\nGoing through DS" + str(ds_cnt + 1))
    
#     # Set model parameters and initial values
#     C = 1000.0
#     m = len(X_train_scaled)
#     initial_alphas = np.zeros(m)
#     initial_b = 0.0

#     # Initialize model
#     model = AAOSVM()

#     tic = time()
#     # Fit model
#     support_vectors, iterations = model.fit(x, y)
#     toc = time()

#     # Support vector count
#     sv_count = support_vectors.shape[0]

#     # Make prediction
#     y_hat = model.predict(x)

#     # Calculate accuracy
#     acc = calc_acc(y, y_hat)

#     print("Support vector count: %d" % (sv_count))
#     print("bias:\t\t%.3f" % (model.b))
#     print("w:\t\t" + str(model.w))
#     print("accuracy:\t%.3f" % (acc))
#     print("Converged after %d iterations" % (iterations))
#     print("Time to train:\t\t" + str(toc - tic))
    
################################################################
################# Test on mock dataset #########################
################################################################
X_train, y = make_blobs(n_samples=1000, centers=2,
                        n_features=2, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train, y)

y[y == 0] = -1

# Set model parameters and initial values
C = 1000.0
m = len(X_train_scaled)
initial_alphas = np.zeros(m)
initial_b = 0.0

# Instantiate model
model = SMOModel(X_train_scaled, y, C,
                 initial_alphas, initial_b, np.zeros(m))

# Initialize error cache
initial_error = model.decision_function(model.X) - model.y
# initial_error = model.decision_function(model.alphas, model.y, model.kernel,
#                                   model.X, model.X, model.b) - model.y
model.errors = initial_error

np.random.seed(0)
model.train()

fig, ax = plt.subplots()
grid, ax = model.plot_decision_boundary(ax)

print(model.alphas.sum())  
################################################################
################# Test on mock dataset #########################
################################################################