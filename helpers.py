import copy
import pandas as pd
import numpy as np


def load_data(fname):
  # we load the data into a dataFrame (df) to return it
  df = pd.read_csv(fname)

  return df


def clean_data(df):
    # replaces infinity and nan values with 0 in a dataFrame
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(0) # or df = df.dropna()

    # gets rid of non-numeric data
    df = df._get_numeric_data()

    return df


# splits the dataFrame into four parts for training (80%) and testing (20%): X_train, y_train, X_test, y_test
def split_data(df):
    mask = np.random.rand(len(df)) < 0.8
    df_train = df[mask]
    df_test = df[~mask]

    X_train = df_train[df_train.columns[:-1]]
    y_train = df_train[df_train.columns[-1]]

    X_test = df_test[df_test.columns[:-1]]
    y_test = df_test[df_test.columns[-1]]

    return X_train, y_train, X_test, y_test