from sklearn.cluster import KMeans
from itertools import product
import numpy as np


# def generating_adversarial_samples(x, selFeatures, df, y=1):
#     """
#     Generates new samples based on previous ones and selected features
#     An important note is that it should be able to deal with samples that got successfully classified as phishing


#     Keyword arguments:
#     x -- an instance (a row)
#     selFeatures -- a list of indexes of the selected features
#     df -- a dataframe that contains the dataset, with labels
#     y -- which samples should be used (+1/legitimate or -1/phishing)
#     """
#     L = pd.DataFrame()
#     L_unique = []
#     gen_samples = []

#     # Gets only the legitimate or phishing samples
#     df = df[df.iloc[:, -1] == y]

#     # Gets the selected features from the dataset
#     for featurePos in selFeatures:
#         L[df.iloc[:, featurePos].name] = df.iloc[:, featurePos].copy()

#     # Gets the unique values from the selected features
#     for col in L:
#         L_unique.append(L[col].unique())

#     # Product possible values to generate all combinations
#     """
#     1 2  ->  1 2 | 3 2 | 3 4 | 1 4
#     3 4      3 4 | 1 4 | 1 2 | 3 2

#     L_pr -- a list of lists, each sublist is a set of values that will be used to generate a new adversarial sample
#     """
#     # test = [[1, 3, 5], [2, 4, 6], [10, 20, 30]]

#     # print(list(product(*test)))

#     # L_pr = list(product(*test))
#     L_pr = list(product(*L_unique))

#     # Generates new samples from the combinations
#     for sample in L_pr:
#         temp = x.copy()
#         i = 0

#         for j in selFeatures:
#             temp[j] = sample[i]
#             i += 1

#         gen_samples.append(temp)

#     return np.array(gen_samples)


def generating_labels(gen_samples: np.ndarray, criteria: int, y=1)->np.array:
    """
    Generates labels for the gen_samples based on 'criteria'.

    Parameters:
        gen_samples (np.ndarray): samples to be labeled.
        criteria (int): how the new samples should be labeled
            0 -> invert labels
            1 -> keep labels
            2 -> generate new labels based on cluster proximity
        y (int): +1 -> for phishing samples/-1 -> for legitimate samples

    Returns:
        gen_labels (np.array): labels for the gen_samples, based on the 'criteria'
    """
    gen_labels = np.ones(len(gen_samples)) * y
    if criteria == 0:
        gen_labels *= -1
    elif criteria == 2:
        if len(gen_labels) > 1:
            km = KMeans(2, random_state=42).fit(gen_samples)
            gen_labels =  np.array(km.labels_)
            gen_labels[gen_labels == 0] = -1
        else:
            labels = [1, -1]
            gen_labels[0] = np.random.choice(labels, 1)

    return gen_labels

def generating_adversarial_samples(
    x: np.array, selFeatures: list, ds: np.ndarray, labels: np.array, criteria: int, y=1
):
    """
    Generates new samples based on previous ones and selected features
    An important note is that it should be able to deal with samples that got successfully classified as phishing
        * does not check if label of 'x' == 'y', i.e., 'x' can have label '-1' and generate samples from '1'

    Parameters:
        x (np.array): an instance, a row from 'ds'
        selFeatures (list): a list of indexes of the selected features to be altered
        ds (np.ndarray): a dataset that contains the 'x' (ds.x)
        labels (np.array): labels of 'ds' (ds.y)
        y (int): which samples should be used
            +1 -> phishing
            -1 -> legitimate
        criteria (int): how the new samples should be labeled
            0 -> invert labels
            1 -> keep labels
            2 -> generate new labels based on cluster proximity

    Returns
        gen_samples (np.ndarray): samples generated from x
        gen_labels (np.array): labels for the generated samples, based on the 'criteria'
    """
    L = []
    L_unique = []
    gen_samples = []

    # Gets only the legitimate or phishing samples
    ds = ds[labels == y]

    # Gets the selected features from the dataset
    for featurePos in selFeatures:
        L.append([ds[:, featurePos].copy()])

    # Gets the unique values from the selected features
    for col in L:
        L_unique.append(np.unique(col))

    # Product possible values to generate all combinations
    """
    1 2  ->  1 2 | 3 2 | 3 4 | 1 4  
    3 4      3 4 | 1 4 | 1 2 | 3 2

    L_pr -- a list of lists, each sublist is a set of values that will be used to generate a new adversarial sample
    """
    # test = np.array([[1, 3, 5], [2, 4, 6], [10, 20, 30]])

    # print(list(product(*test)))

    # L_pr = list(product(*test))
    L_pr = list(product(*L_unique))
    # /Product possible values to generate all combinations

    # Generates new samples from the combinations
    for sample in L_pr:
        temp = x.copy()
        i = 0

        for j in selFeatures:
            temp[j] = sample[i]
            i += 1

        gen_samples.append(temp)

    return np.array(gen_samples), np.array(generating_labels(gen_samples, criteria, y))


def cost(a, b):
    """
    Euclidean distance between two pandas.Series or np.arrays
    """

    return np.linalg.norm(a - b)


# print(ds1.head())
# print(generating_adversarial_samples(ds1.iloc[0, :], [0,2,4], ds1))
# print(len(generating_adversarial_samples(ds1.iloc[0, :], [0,2,4], ds1)))
# generating_adversarial_samples(ds1.iloc[0, :], [0,2,4], ds1)
# print(ds1.iloc[0, :])
# print(df.dtypes)
