from scipy.io import arff
import pandas as pd
import numpy as np

data = arff.loadarff('Datasets\PhishingData.arff')  # all attributes in range [-1, 1]
# data = arff.loadarff('Datasets\TrainingDataset.arff')  # all attributes in range [-1, 1]
df = pd.DataFrame(data[0], dtype=np.int8)