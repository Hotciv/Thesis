from reading_datasets import *
import pickle
import glob

# print(glob.glob('Results/*/*.pkl'))
# print(glob.glob('Results/*/'))

# results = glob.glob('Results/*/')
results = glob.glob('Results, new/*/')

print("Please choose which results do you wish to look into:")
for i, r in enumerate(results):
    print(i, '->', r)
i = int(input())

# each of these will contain one passage of crossvalidation
print('Models to be loaded from crossvalidation:')
# for cv in glob.glob(results[i] + "*.pkl"):
for cv in glob.glob(results[i] + "Decision Tree_ds1_std_0.pkl"):
    print(cv)
    
    with open(cv, 'rb') as f:
        model = []
        try:
            while True:
                model.append(pickle.load(f))
        except EOFError:
            print('Loaded all the models from this cross validation')

    print(len(model))
    print(model)