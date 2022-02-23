# from accelerate import profiler
import cProfile
import pstats

from sklearn.model_selection import KFold
from aux_functions import cross_validate

from reading_datasets import *
from AAOSVM import *

from csv import writer
from time import time
import pickle

def profile_test():
    datasets = to_dataset([2])

    np.random.seed(0)

    cv = 5  # cross validation
    kf = KFold(n_splits=cv)

    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        print("\n\nGoing through DS" + str(ds_cnt + 1))

        # Set model parameters and initial values
        C = 100.0
        m = 2
        initial_alphas = np.zeros(m)
        initial_b = 0.0
        initial_w = np.zeros(len(ds[0][0]))
        initial_Psi = 0.0

        # Slidding window
        Sx = ds[0][0]
        Sy = ds[1][0]
        Sx = np.append([Sx], [ds[0][1]], 0)
        Sy = np.append([Sy], [ds[1][1]], 0)

        # Initialize model
        model = AAOSVM(Sx, Sy, C, initial_alphas, initial_b, np.zeros(m))
        
        # Initialize error cache
        initial_error = model.decision_function(model.X) - model.y
        model.errors = initial_error
        
        model.w = initial_w

        # X = ds[0]
        # Y = ds[1]
        start_time = time()
        ACCs, TPRs, F1s, _ = cross_validate(model, ds[0], ds[1], 'AAOSVM', 'AAOSVM', random_state=ds_cnt)
        finish = time() - start_time

        print('finished', finish)

        print("Support vector count: %d" % (np.count_nonzero(model.alphas)))
        print("bias:\t\t%.3f" % (model.b))
        print("w:\t\t" + str(model.w))

aux = cProfile.run('profile_test()', sort='cumtime')
print(aux)