from itertools import product
from reading_datasets import *
# import pandas as pd


def generating_adversarial_samples(x, selFeatures, df, y=1):
    '''Generates new samples based on previous ones and selected features
    An important note is that it should be able to deal with samples that got successfully classified as phishing


    Keyword arguments:
    x -- an instance (a row)
    selFeatures -- a list of indexes of the selected features
    df -- a dataframe that contains the dataset
    y -- which samples should be used (+1/legitimate or -1/phishing)
    '''
    L = pd.DataFrame()
    L_unique = []
    genSamples = []

    # Gets only the legitimate or phishing samples
    df = df[df.iloc[:, -1] == y]

    # Gets the selected features from the dataset
    for featurePos in selFeatures:
        L[df.iloc[:, featurePos].name] = df.iloc[:, featurePos].copy()

    # Gets the unique values from the selected features
    for col in L:
        L_unique.append(L[col].unique())

    # Product possible values to generate all combinations
    '''
    1 2  ->  1 2 | 3 2 | 3 4 | 1 4  
    3 4      3 4 | 1 4 | 1 2 | 3 2

    L_pr -- a list of lists, each sublist is a set of values that will be used to generate a new adversarial sample
    '''
    # test = [[1, 3, 5], [2, 4, 6, 8], [10, 20]]

    # print(list(product(*test)))
    
    # L_pr = list(product(*test))
    L_pr = list(product(*L_unique))

    # Generates new samples from the combinations
    for sample in L_pr:
        temp = x.copy()
        i = 0
        
        for j in selFeatures:
            temp[j] = sample[i]
            i += 1
        
        genSamples.append(temp)

    return genSamples


def cost(a, b):
    '''
    Euclidean distance between two pandas.Series
    '''

    return (np.linalg.norm(a-b))

# print(ds1.head())
# print(generating_adversarial_samples(ds1.iloc[0, :], [0,2,4], ds1))
# print(len(generating_adversarial_samples(ds1.iloc[0, :], [0,2,4], ds1)))
# generating_adversarial_samples(ds1.iloc[0, :], [0,2,4], ds1)
# print(ds1.iloc[0, :])
# print(df.dtypes)