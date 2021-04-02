from scipy.io import arff
import pandas as pd
import numpy as np

# data = arff.loadarff('Datasets\PhishingData.arff')
# ds1 = pd.DataFrame(data[0], dtype=np.int8)
'''
    All attributes in range [-1, 1]
    Result: -1 - phishing
             0 - suspicious
             1 - legitimate
'''

# data = arff.loadarff('Datasets\TrainingDataset.arff')
# ds2 = pd.DataFrame(data[0], dtype=np.int8)
'''
    All attributes in range [-1, 1]
    Result not clear, will adopt this convention:
            -1 - phishing
             1 - legitimate
'''

# data = arff.loadarff('Datasets\h3cgnj8hft-1\Phishing_Legitimate_full.arff')
# ds3 = pd.DataFrame(data[0], dtype=np.int8)
'''
    Some attributes are > 1, some are between 0 and 1, inclusive
    CLASS_LABEL not clear, will adopt this convention: 
             0 - phishing
             1 - legitimate
'''

# ds4 = pd.read_csv('Datasets/72ptz43s9v-1/dataset_full.csv')
# ds4_sub = pd.read_csv('Datasets/72ptz43s9v-1/dataset_small.csv')
'''
    Some attributes are > 1, some are between 0 and 1, inclusive
    phishing: 0 - legitimate
              1 - phishing
'''