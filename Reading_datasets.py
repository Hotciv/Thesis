from scipy.io import arff
import pandas as pd
import numpy as np

# data = arff.loadarff('Datasets\PhishingData.arff')
data = arff.loadarff('Datasets\TrainingDataset.arff')
df = pd.DataFrame(data[0], dtype=np.int8)

print(df.head())
print(df.dtypes)