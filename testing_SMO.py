from reading_datasets import *
from time import time
from SVM_w_SMO_master.SVM import *

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

# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    print("\n\nGoing through DS" + str(ds_cnt + 1))
    
    # Initialize model
    model = SVM()

    tic = time()
    # Fit model
    support_vectors, iterations = model.fit(x, y)
    toc = time()

    # Support vector count
    sv_count = support_vectors.shape[0]

    # Make prediction
    y_hat = model.predict(x)

    # Calculate accuracy
    acc = calc_acc(y, y_hat)

    print("Support vector count: %d" % (sv_count))
    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (acc))
    print("Converged after %d iterations" % (iterations))
    print("Time to train:\t\t" + str(toc - tic))
    

    
