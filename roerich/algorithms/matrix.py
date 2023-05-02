import pandas as pd
import numpy as np
import torch
import roerich
from itertools import combinations


class MatrixScore():

    def __init__(self, cpd):
        """
        Algorithm for calculating pairwise scores

        Parameters:
        -----------
        cpd: roerich.algorithms.ChangePointDetectionClassifier
            Change point detection algorithm 
        """

        self.cpd = cpd


    def union_score(self, data):
        """
        Calculates matrixcies of scores by pairwise adding features
        
        Parameters:
        -----------
        data: numpy.array
            A broader time-step interval

        Returns:
        --------
        matrix: numpy.array
            3d array - list of score matricies 
        """

        if type(data) == pd.DataFrame: data = data.to_numpy()

        assert type(data) == np.ndarray, "'data' argument must be numpy array or pandas dataframe."
        assert data.shape[1] > 0, "'data' must contain at least 1 feature"

        single_FI = {}
        for feature in range(data.shape[1]):
            single_FI[feature], _ = self.cpd.predict(data[:, feature])

        pairs = list(combinations(np.arange(data.shape[1]), 2))

        paired_FI = {}
        for pair in pairs:
            paired_FI[pair], _ = self.cpd.predict(data[:, list(pair)])
            
            
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
        
        matrix = np.swapaxes(matrix.numpy(), 0, 2)
        
        return matrix


    def exclude_score(self, data):
        """
        Calculates matrixcies of scores by pairwise deleting features
        
        Parameters:
        -----------
        data: numpy.array
            A broader time-step interval

        Returns:
        --------
        matrix: numpy.array
            3d array - list of score matricies 
        """

        if type(data) == pd.DataFrame: data = data.to_numpy()

        assert type(data) == np.ndarray, "'data' argument must be numpy array or pandas dataframe."
        assert data.shape[1] > 0, "'data' must contain at least 1 feature"

        single_FI = {}
        for feature in range(data.shape[1]):
            single_FI[feature], _ = self.cpd.predict(data[:, feature])

        pairs = list(combinations(np.arange(data.shape[1]), 2))

        paired_FI = {}
        for pair in pairs:
            paired_FI[pair], _ = self.cpd.predict(np.delete(data, list(pair), 1))
            
            
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
        
        matrix = np.swapaxes(matrix.numpy(), 0, 2)

        return matrix


