#!/usr/bin/env python

import pandas as pd
import numpy as np
import math


def readData(filename = "Queens.csv"):
    data = pd.read_csv(filename)
    print("Data loading finished.")
    return data.values[:,0:-1]


def splitData(n):
    """
    Split data index into train index, validation, test index with a ratio 8:1:1.
    input:
    n: number of observations
    output:
    train: one-dimensional array
    test: one-dimensional array
    """
    split1 = math.floor(n * 0.8)
    split2 = math.floor(n * 0.9)
    permutated_index = np.random.permutation(np.arange(n))
    train = permutated_index[:split1]
    validation = permutated_index[split1:split2]
    test = permutated_index[split1:split2]
    print("Data splitting finished.")
    return train, validation, test