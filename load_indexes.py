from aux_functions import dataset_split
from reading_datasets import *
from numpy import where
from glob import glob
import pickle
from sklearn.model_selection import KFold
from inspect import getmembers, isroutine
import matplotlib.pyplot as plt
from adversarial_samples import cost

datasets, names = to_dataset([2])
ds = datasets[0]
X, y = ds

# results = glob('Results/*/')
results = glob("Results, new/*/")

print("Please choose which results do you wish to look into:")
for i, r in enumerate(results):
    print(i, "->", r)
i = int(input())

# each of these will contain one passage of crossvalidation
print("Models to be loaded from crossvalidation:")
for cv in glob(results[i] + "Linear SVM_ds3_std_0.pkl"):
    print(cv)

    with open(cv, 'rb') as f:
        model = []
        try:
            for j in range(5):
                model.append(pickle.load(f))
        except EOFError:
            print('Loaded all the models from this cross validation')
        finally:
            for j in range(5):
                # print(where(model[j].support_vectors_ == ds))  #TODO: continue this search?
                # print(model[j].support_)
                print(model[j].support_vectors_.shape)
                input()
                # print(y[model[j].support_])
                y_ = y[model[j].support_]
                print(len(y_[y_ == 1]))
                input()

# input()
# for cv in glob(results[i] + "RBF SVM_ds*_std_*.pkl"):
#     print(cv)

#     with open(cv, 'rb') as f:
#         model = []
#         try:
#             for j in range(5):
#                 model.append(pickle.load(f))
#         except EOFError:
#             print('Loaded all the models from this cross validation')
#         finally:
#             for j in range(5):
#                 print(model[j].support_vectors_.shape)

# # input()
# # for cv in glob(results[i] + "inc_ds*_OSVM_*.pkl"):
# #     print(cv)

# #     with open(cv, 'rb') as f:
# #         model = []
# #         try:
# #             for j in range(5):
# #                 model.append(pickle.load(f))
# #         except EOFError:
# #             print('Loaded all the models from this cross validation')
# #         finally:
# #             for j in range(5):
# #                 print(model[j].get_params())

# # beginning of a KNN ./
# for cv in glob(results[i] + "indexes - _" + names[0] + "_0_0.pkl"):
#     with open(cv, "rb") as f:
#         final_indexes = None
#         try:
#             while True:
#                 final_indexes = pickle.load(f)
#         except EOFError:
#             print("Indexes locked and loaded")
#         # print(final_indexes)
#         # print(final_indexes[0] + final_indexes[1])
#         # print(y[final_indexes[0] + final_indexes[1]])
#         # print(len(y_[y_ == 1]))
#         # print(X_[y_ == 1])
#         y_ = y[final_indexes[0] + final_indexes[1]]
#         X_ = X[final_indexes[0] + final_indexes[1]]
#         sz = len(X_[y_ == 1])
#         the_200 = set()
#         for j, sample in enumerate(X[y == 1]):
#             for k, sample_ in enumerate(X_[y_ == 1]):
#                 c = cost(sample, sample_)
#                 if c == 0 and len(the_200) < sz:
#                     print(j, k, c, len(the_200))
#                     the_200.add(j)
#                     break
#                 if len(the_200) == sz:
#                     break
#             if len(the_200) == sz:
#                     break
#         # print(the_200)
#         # input()

#         rad = 1
#         dist = 2
#         while len(the_200) < 200:
#             for j, sample in enumerate(X[y == 1]):
#                 for k, sample_ in enumerate(X_[y_ == 1]):
#                     c = cost(sample, sample_)
#                     if c > 0 and c <= rad and len(the_200) < 200:
#                         print(j, k, rad, c, len(the_200))
#                         the_200.add(j)
#                         break
#                     if len(the_200) == 200:
#                         break
#                 if len(the_200) == 200:
#                         break
#             rad = np.sqrt(dist)
#             dist += 1
#         print(the_200)

input()
for cv in glob(results[i] + "AAOSVM_" + names[0] + "_AAOSVM_0.pkl"):
    print(cv)

    with open(cv, "rb") as f:
        model = []
        try:
            for j in range(5):
                model.append(pickle.load(f))
        except EOFError:
            print("Loaded all the models from this cross validation")
        finally:
           # trying to retrieve the support vectors
           for j in range(5):
               print(model[j].errors[0:11])
               print(model[j].y[0:10])
    #             kf = KFold(n_splits=5, random_state=42, shuffle=True)

    #             for train_index, test_index in kf.split(X, y):
    #                 X_partial, X_hold = X[train_index], X[test_index]
    #                 y_partial, y_hold = y[train_index], y[test_index]

    #                 # Slidding window
    #                 Sx = X_partial[0:100]
    #                 Sy = y_partial[0:100]

    #                 sz = len(y_partial)
    #                 for i in range(100, sz, 1):
    #                     # finding the saved values
    #                     errors = model[j].decision_function(Sx) - Sy
    #                     # weights = np.sum(Sy * model[j].alphas * Sx.T, axis=1)

    #                     if np.all(model[j].errors == errors):
    #                     # if np.all(model[j].w == weights):
    #                         break

    #                     if i % (sz // 100) == 0:
    #                         print("reached final {}".format(i))

    #                     x = X_partial[i]
    #                     Y = y_partial[i]

    #                     # if reached the limit of the window
    #                     # i.e. window is full and now will move
    #                     if Sy.shape[0] >= 100:
    #                         Sx = Sx[1:]
    #                         Sy = Sy[1:]

    #                     # Adding instance to the window
    #                     Sx = np.append(Sx, [x], 0)
    #                     Sy = np.append(Sy, [Y], 0)

                # # see attributes
                # attributes = getmembers(model[j], lambda a: not (isroutine(a)))
                # for a in attributes:
                #     if not (a[0].startswith("__") and a[0].endswith("__")):
                #         if not (isinstance(a[1], int) or isinstance(a[1], float)):
                #             if a[0] != "clusters":
                #                 print(a[0], len(a[1]))
                #         elif isinstance(a[1], int) or isinstance(a[1], float):
                #             print(a)
                # input()


# # for cv in glob(results[i] + "indexes - ds1, ds2, ds3, ds4, ds5.pkl"):
# for cv in glob(results[i] + "indexes/indexes - ds*.pkl"):
#     print(cv)

#     with open(cv, 'rb') as f:
#         try:
#             while True:
#                 print(pickle.load(f))
#         except EOFError:
#             print('Loaded all the indexes')
