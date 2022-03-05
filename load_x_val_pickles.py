import pickle

model = []

with open('Linear SVM_std_0' + '.pkl', 'rb') as f:
    try:
        while True:
            model.append(pickle.load(f))
    except EOFError:
        print('Loaded all the models from the cross validation')

print(len(model))