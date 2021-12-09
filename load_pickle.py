import pickle
import numpy as np

filename = input('filename:\n')
with open(filename + '.pkl', 'rb') as f:
    try:
        # aux = pickle.load(f)
        # while isinstance(aux, np.ndarray):
        while True:
            aux = pickle.load(f)
            print(aux)
    except EOFError:
        print("EOF")