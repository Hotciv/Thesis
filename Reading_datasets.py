# from numpy.core.shape_base import block
# import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
import numpy as np

'''
    All classes will be relabeled as follows:
    +1 -> phishing instances
    -1 -> legitimate instances

    When generating adversarial samples, 200 phishing instances will be reserved from each dataset
'''

# ds1 = pd.read_csv('Datasets/72ptz43s9v-1/dataset_full.csv')
ds1_sub = pd.read_csv('Datasets/72ptz43s9v-1/dataset_small.csv')
ds1_sub["phishing"].replace({0: -1}, inplace=True)  # test column name
'''
    58645 instances x 111 attributes
    Some attributes are > 1, some are between 0 and 1, inclusive
    phishing: 0 - legitimate
              1 - phishing
'''


data = arff.loadarff('Datasets\TrainingDataset.arff')
ds2 = pd.DataFrame(data[0], dtype=np.int8)
ds2["Result"].replace({1: -1, -1: 1}, inplace=True)
'''
    11055 instances x 30 attributes
    All attributes in range [-1, 1]
    Result not clear, will adopt this convention:
            -1 - phishing
             1 - legitimate
'''

data = arff.loadarff('Datasets\PhishingData.arff')
ds3 = pd.DataFrame(data[0], dtype=np.int8)
# aux = ds1[ds1["Result"] == 0]
# aux.loc["Result"] = 1
ds3["Result"].replace({0: 1}, inplace=True)
ds3["Result"].replace({1: -1, -1: 1}, inplace=True)
'''
    1353 instances x 9 attributes
    All attributes in range [-1, 1]
    Result original values: 
            -1 - phishing
             0 - suspicious
             1 - legitimate

    Shirazi et al. assumed the suspicious as legitimate
'''

# fig, ax = plt.subplots(1,1)
# bins = np.arange(-1,3,1)
# ax.set_xlabel('SFH')
# ax.set_ylabel('Number of instances')
# ax.hist(ds3['SFH'], bins=bins, align='left', rwidth=.5)
# ax.set_xticks(bins[:-1])
# plt.show()

# fig, ax = plt.subplots(1,1)
# ax.set_xlabel('SFH')
# ax.set_ylabel('Number of instances')
# ax.hist([ds3['SFH'][ds3['Result'] == 1], ds3['SFH'][ds3['Result'] == -1]], bins=bins, align='left', rwidth=.5)
# ax.set_xticks(bins[:-1])
# plt.show()

# fig, ax = plt.subplots(1,1)
# ax.set_xlabel('popUpWidnow')
# ax.set_ylabel('Number of instances')
# ax.hist(ds3['popUpWidnow'], bins=bins, align='left', rwidth=.5)
# ax.set_xticks(bins[:-1])
# plt.show()

# fig, ax = plt.subplots(1,1)
# ax.set_xlabel('popUpWidnow')
# ax.set_ylabel('Number of instances')
# ax.hist([ds3['popUpWidnow'][ds3['Result'] == 1],ds3['popUpWidnow'][ds3['Result'] == -1]], bins=bins, align='left', rwidth=.5)
# ax.set_xticks(bins[:-1])
# plt.show()

data = arff.loadarff('Datasets\h3cgnj8hft-1\Phishing_Legitimate_full.arff')
ds4 = pd.DataFrame(data[0], dtype=np.int8)
ds4["CLASS_LABEL"].replace({0: 1, 1: -1}, inplace=True)
ds4 = ds4.sample(frac=1, random_state=42).reset_index(drop=True)
'''
    10000 instances x 48 attributes
    Some attributes are > 1, some are between 0 and 1, inclusive
    CLASS_LABEL not clear, will adopt this convention: 
             0 - phishing
             1 - legitimate
'''

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# ds5 = scaler.fit_transform(ds4)
# '''
#     ds4 after passing through minmax normalization
# '''
