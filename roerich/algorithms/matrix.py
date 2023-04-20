import pandas as pd
import numpy as np
import torch
import roerich
from itertools import combinations


def MatrixUnionPairImportance(data, cpd, check=True):
    if type(data) == pd.DataFrame:
        data = data.to_numpy()

    assert type(data) == np.ndarray, "'data' argument must be numpy array or pandas dataframe."
    assert data.shape[1] > 0, "'data' must contain at least 1 feature"

    ROERICH_ALGOS = [
        roerich.algorithms.cpdc.ChangePointDetectionBase,
        roerich.algorithms.cpdc.ChangePointDetectionClassifier,
        roerich.algorithms.cpdc.ChangePointDetectionRuLSIF,
        roerich.algorithms.OnlineNNClassifier,
        roerich.algorithms.OnlineNNRuLSIF
    ]
        
    if check: assert type(cpd) in ROERICH_ALGOS, "Sorry, this algo either not Roerich's or doesn't have score function"

    single_FI = {}
    for feature in range(data.shape[1]):
        single_FI[feature], _ = cpd.predict(data[:, feature])

    pairs = list(combinations(np.arange(data.shape[1]), 2))

    paired_FI = {}
    for pair in pairs:
        paired_FI[pair], _ = cpd.predict(data[:, list(pair)])
        
        
    m, n = data.shape

    matrix = torch.empty(n, n, m)

    for feature_1 in range(n):
        for feature_2 in range(n): 

            if feature_1 != feature_2:
                pair = None
                if (feature_1, feature_2) in pairs: pair = (feature_1, feature_2)
                if (feature_2, feature_1) in pairs: pair = (feature_2, feature_1)
                c = paired_FI[pair]

                a = single_FI[feature_1]
                b = single_FI[feature_2]

                matrix[feature_1, feature_2, :] = torch.tensor(c)
            else:
                matrix[feature_1, feature_2, :] = torch.tensor(single_FI[feature_1])
    
    return matrix


def MatrixExcludePairImportance(data, cpd, check=True):
    if type(data) == pd.DataFrame:
        data = data.to_numpy()

    assert type(data) == np.ndarray, "'data' argument must be numpy array or pandas dataframe."
    assert data.shape[1] > 0, "'data' must contain at least 1 feature"

    single_FI = {}
    for feature in range(data.shape[1]):
        single_FI[feature], _ = cpd.predict(data[:, feature])

    pairs = list(combinations(np.arange(data.shape[1]), 2))

    paired_FI = {}
    for pair in pairs:
        paired_FI[pair], _ = cpd.predict(np.delete(data, list(pair), 1))
        
        
    m, n = data.shape

    matrix = torch.empty(n, n, m)

    for feature_1 in range(n):
        for feature_2 in range(n): 

            if feature_1 != feature_2:
                pair = None
                if (feature_1, feature_2) in pairs: pair = (feature_1, feature_2)
                if (feature_2, feature_1) in pairs: pair = (feature_2, feature_1)
                c = paired_FI[pair]

                a = single_FI[feature_1]
                b = single_FI[feature_2]

                matrix[feature_1, feature_2, :] = torch.tensor(c)
            else:
                matrix[feature_1, feature_2, :] = torch.tensor(single_FI[feature_1])
    
    return matrix


