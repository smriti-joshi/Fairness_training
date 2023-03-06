import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

class Preprocessor:
    def __init__(self):
        self.drop = None
        self.high_variance_selector = VarianceThreshold(threshold=0.15)
        self.correlation_threshold = 0.9

    def set_correlation_threshold(self, threshold):
        self.correlation_threshold = threshold

    def drop_correlated_features(self, features):

        df = pd.DataFrame(features)

        # Create correlation matrix
        corr_matrix = df.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where((np.triu(np.ones(corr_matrix.shape), k=1) + np.tril(np.ones(corr_matrix.shape), k=-1)).astype(bool))

        # Find features with correlation greater than 0.95
        self.drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]

        # Drop features 
        df.drop(self.drop, axis=1, inplace=True)

        return df
    
    def preprocess_train(self, features):
        return self.high_variance_selector.fit_transform(self.drop_correlated_features(features))
    
    def preprocess_val(self, features):
        df = pd.DataFrame(features)
        df.drop(self.drop, axis=1, inplace=True)
        return df.iloc[:, self.high_variance_selector.get_support(indices=True)].to_numpy()