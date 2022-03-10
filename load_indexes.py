import pickle
from glob import glob
from numpy import where

# results = glob('Results/*/')
results = glob('Results, new/*/')

print("Please choose which results do you wish to look into:")
for i, r in enumerate(results):
    print(i, '->', r)
i = int(input())

# each of these will contain one passage of crossvalidation
print('Models to be loaded from crossvalidation:')
for cv in glob(results[i] + "Linear SVM_ds*_std_*.pkl"):
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
                print(model[j].support_vectors_.shape)

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

# input()
# for cv in glob(results[i] + "AAOSVM_ds*_AAOSVM_*.pkl"):
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
#                 print(where(model[j].alphas > 0)[0])


# # for cv in glob(results[i] + "indexes - ds1, ds2, ds3, ds4, ds5.pkl"):
# for cv in glob(results[i] + "indexes/indexes - ds*.pkl"):
#     print(cv)

#     with open(cv, 'rb') as f:
#         try:
#             while True:
#                 print(pickle.load(f))
#         except EOFError:
#             print('Loaded all the indexes')