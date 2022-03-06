# from numpy.core.shape_base import block
# import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
import numpy as np

"""
    All classes will be relabeled as follows:
    +1 -> phishing instances
    -1 -> legitimate instances

    When generating adversarial samples, 200 phishing instances will be reserved from each dataset
"""

dataframes = []

# ds1 (AKA ds5):
aux = pd.read_csv('Datasets/72ptz43s9v-1/dataset_full.csv')
aux["phishing"].replace({0: -1}, inplace=True)
dataframes.append(aux)
"""
    TODO: change
    58645 instances x 111 attributes
    Some attributes are > 1, some are between 0 and 1, inclusive
    phishing: 0 - legitimate
              1 - phishing
"""

# ds1_sub (AKA ds1):
aux = pd.read_csv("Datasets/72ptz43s9v-1/dataset_small.csv")
aux["phishing"].replace({0: -1}, inplace=True)
dataframes.append(aux)
"""
    58645 instances x 111 attributes
    Some attributes are > 1, some are between 0 and 1, inclusive
    phishing: 0 - legitimate
              1 - phishing
"""


# ds2
data = arff.loadarff("Datasets\TrainingDataset.arff")
aux = pd.DataFrame(data[0], dtype=np.int8)
aux["Result"].replace({1: -1, -1: 1}, inplace=True)
dataframes.append(aux)
"""
    11055 instances x 30 attributes
    All attributes in range [-1, 1]
    Result not clear, will adopt this convention:
            -1 - phishing
             1 - legitimate
"""

# ds3
data = arff.loadarff("Datasets\PhishingData.arff")
aux = pd.DataFrame(data[0], dtype=np.int8)
aux["Result"].replace({0: 1}, inplace=True)
aux["Result"].replace({1: -1, -1: 1}, inplace=True)
dataframes.append(aux)
"""
    1353 instances x 9 attributes
    All attributes in range [-1, 1]
    Result original values: 
            -1 - phishing
             0 - suspicious
             1 - legitimate

    Shirazi et al. assumed the suspicious as legitimate
"""

# ds4
data = arff.loadarff("Datasets\h3cgnj8hft-1\Phishing_Legitimate_full.arff")
aux = pd.DataFrame(data[0], dtype=np.int8)
aux["CLASS_LABEL"].replace({0: 1, 1: -1}, inplace=True)
aux = aux.sample(frac=1, random_state=42).reset_index(drop=True)
dataframes.append(aux)
"""
    10000 instances x 48 attributes
    Some attributes are > 1, some are between 0 and 1, inclusive
    CLASS_LABEL not clear, will adopt this convention: 
             0 - phishing
             1 - legitimate
"""

# sending ds5 to it's place
dataframes = dataframes[1:] + [dataframes[0]]

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# ds5 = scaler.fit_transform(ds4)
# '''
#     ds4 after passing through minmax normalization
# '''


def to_dataset(df_numbers=[0, 1, 2, 3, 4]):
    """
    Transforms Pandas DataFrames into numpy datasets which consists of tuples of 'x' and 'y' where
        x (np.ndarray): an array of instances and each instance has a number of features.
        y (np.array): an array of labels.

    As a bonus, also get the correct names to include in filename.

    Parameters:
        df_numbers (list): a list of DataFrames indexes to be turned into "numpy datasets" (described above).

    Returns:
        datasets (list): a list of "numpy datasets" (described above).
        name (str): a str to be inserted into the filename.
    """

    # DataFrames -> datasets
    datasets = []
    for index in df_numbers:
        ds = dataframes[index]
        y = ds.pop(ds.columns.tolist()[-1])
        y = y.to_numpy()
        x = ds.to_numpy()
        datasets.append((x, y))

    # name
    names = ["ds1", "ds2", "ds3", "ds4", "ds5"]
    name = ''
    for i in df_numbers:
        name = name + names[i] + ', '
    name = name[:-2]

    return datasets, name


def get_mock_datasets():
    """
    Gets linearly separable dataset in format of numpy datasets which consists of tuples of 'x' and 'y' where
        x (np.ndarray): an array of instances and each instance has a number of features.
        y (np.array): an array of labels.

    Also creates a duplicate of this dataset with two outliers at the end.

    Returns:
        datasets (list): a list of two "numpy datasets" (described above) from mock data.
    """
    from sklearn.datasets import make_blobs, make_circles
    from sklearn.preprocessing import StandardScaler

    datasets = []
    X_train, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y)

    y[y == 0] = -1

    # Add an outlier
    X_outlier = np.append(X_train_scaled, [0.1, 0.1])
    X_outlier = X_outlier.reshape(X_train.shape[0] + 1, X_train.shape[1])
    y_outlier = np.append(y, 1)

    # Add a second outlier, so the first one is showed
    X_outlier = np.append(X_outlier, [0.1, 0.1])
    X_outlier = X_outlier.reshape(X_train.shape[0] + 2, X_train.shape[1])
    y_outlier = np.append(y_outlier, -1)

    # Adding mock datasets
    datasets.append((X_train_scaled, y))
    datasets.append((X_outlier, y_outlier))

    return datasets


def dataset_normalization(datasets):
    """
    Function to normalize datasets using MinMaxScaler.

    Parameters:
        datasets (list): list of np.ndarray datasets without the labels.

    Returns:
        normalized_datasets (list): a list of "numpy datasets" (described above).
    """
    from sklearn.preprocessing import MinMaxScaler

    normalized_datasets = []

    scaler = MinMaxScaler()
    for ds in datasets:
        normalized_datasets.append(scaler.fit_transform(ds))

    return normalized_datasets