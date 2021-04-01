from reading_datasets import *
import pandas as pd


def generating_adversarial_samples(x, selFeatures):
    '''Generates new samples based on previous ones and selected features

    Keyword arguments:
    x -- an instance (a row)
    selFeatures -- a list of indexes of the selected features
    '''
    L = pd.DataFrame()

    for featurePos in selFeatures:
        L[df.iloc[:, featurePos].name] = df.iloc[:, featurePos].copy()

    # Product possible values to generate all combinations
    return L


print(df.head())
print(generating_adversarial_samples(None, [0,2,4]))
# print(df.dtypes)