from scipy.io import arff
import pandas as pd
import numpy as np

data = arff.loadarff('Datasets\PhishingData.arff')
ds1 = pd.DataFrame(data[0], dtype=np.int8)
'''
    all attributes in range [-1, 1]
    Result: -1 - phishing
             0 - suspicious
             1 - legitimate
'''
# data = arff.loadarff('Datasets\TrainingDataset.arff')  # all attributes in range [-1, 1]
# ds2 = pd.DataFrame(data[0], dtype=np.int8)