import numpy as np
from numpy.random import shuffle
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor


class RandomForestMSE:
    def __init__(
        self, n_estimators: int, max_depth=None, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

        self.estimators: list[DecisionTreeRegressor] = []
        self.feature_subsample_size = feature_subsample_size
        for i in range(n_estimators):
            tree = DecisionTreeRegressor(max_depth=max_depth, criterion='squared_error', *trees_parameters)
            self.estimators.append(tree)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        X_val : numpy ndarray
            Array of size n_val_objects, n_features
        y_val : numpy ndarray
            Array of size n_val_objects
        """
        
        _, q = X.shape
        if self.feature_subsample_size is None:
            max_features = int(q / 3)
        else:
            max_features = self.feature_subsample_size
        seq: np.ndarray = np.arange(q).astype('int')
        self.list_indexes = []
        for estimator in self.estimators:
            shuffle(seq)
            indexes: np.ndarray = seq[:max_features]
            self.list_indexes.append(indexes.copy())
            estimator.fit(X[:, indexes], y)

    def predict(self, X: np.ndarray):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        y = np.zeros(X.shape[0])
        
        for estimator, index in zip(self.estimators, self.list_indexes):
            y += estimator.predict(X[:, index])
            
        return y / len(self.estimators)
            

class GradientBoostingMSE:
    def __init__(
        self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
        **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """

    def fit(self, X, y, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """