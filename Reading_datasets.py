from scipy.io import arff
import pandas as pd
import numpy as np

'''
    All classes will be relabeled as follows:
    +1 -> phishing instances
    -1 -> legitimate instances

    When generating adversarial samples, 200 phishing instances were reserved from each dataset
'''

# ds1 = pd.read_csv('Datasets/72ptz43s9v-1/dataset_full.csv')
ds1_sub = pd.read_csv('Datasets/72ptz43s9v-1/dataset_small.csv')
ds1_sub["phishing"].replace({0: -1}, inplace=True)  # test column name
'''
    Some attributes are > 1, some are between 0 and 1, inclusive
    phishing: 0 - legitimate
              1 - phishing
'''


data = arff.loadarff('Datasets\TrainingDataset.arff')
ds2 = pd.DataFrame(data[0], dtype=np.int8)
ds2["Result"].replace({1: -1, -1: 1}, inplace=True)
'''
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
    All attributes in range [-1, 1]
    Result: -1 - phishing
             0 - suspicious
             1 - legitimate

    Shirazi et al. assumed the suspicious as legitimate
'''

data = arff.loadarff('Datasets\h3cgnj8hft-1\Phishing_Legitimate_full.arff')
ds4 = pd.DataFrame(data[0], dtype=np.int8)
ds4["CLASS_LABEL"].replace({0: 1, 1: -1}, inplace=True)
'''
    Some attributes are > 1, some are between 0 and 1, inclusive
    CLASS_LABEL not clear, will adopt this convention: 
             0 - phishing
             1 - legitimate
'''

'''
    Creating a dataset from a dataframe:

    target = df.pop('target')
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))

    ################################################################################################################################

    According to documentation of Tensorflow, the best way of preserving the columns of the dataframe might be by making a dictionary:
    "The easiest way to preserve the column structure of a pd.DataFrame when used with tf.data
     is to convert the pd.DataFrame to a dict, and slice that dictionary."

    dict_slices = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values)).batch(16)

'''