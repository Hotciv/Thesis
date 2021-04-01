from reading_datasets import *
import pandas as pd


def generating_adversarial_samples(x, selFeatures, df):
    '''Generates new samples based on previous ones and selected features

    Keyword arguments:
    x -- an instance (a row)
    selFeatures -- a list of indexes of the selected features
    df -- a dataframe that contains the dataset
    '''
    L = pd.DataFrame()

    for featurePos in selFeatures:
        L[df.iloc[:, featurePos].name] = df.iloc[:, featurePos].copy()

    # Product possible values to generate all combinations
    return L


print(ds1.head())
print(generating_adversarial_samples(ds1.iloc[0, :], [0,2,4], ds1))
# print(df.dtypes)