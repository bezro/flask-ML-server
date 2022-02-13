import numpy as np
from numpy.random import shuffle
from scipy.optimize import minimize_scalar
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets


class RandomForestMSE:
    def __init__(
        self, n_estimators=10, max_depth=None, feature_subsample_size=None,
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

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

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
        

        self.estimators: list[DecisionTreeRegressor] = []
        _, q = X.shape
        if self.feature_subsample_size is None:
            max_features = int(q / 3)
        else:
            max_features = self.feature_subsample_size
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                criterion='squared_error', 
                *self.trees_parameters)
            self.estimators.append(tree)
        seq: np.ndarray = np.arange(q).astype('int')
        self.list_indexes = []
        for estimator in self.estimators:
            shuffle(seq)
            indexes: np.ndarray = seq[:max_features]
            self.list_indexes.append(indexes.copy())
            estimator.fit(X[:, indexes], y)

    def predict(self, X: np.ndarray) -> np.ndarray:
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
    
    def get_params(self, deep=False):
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'feature_subsample_size': self.feature_subsample_size
        }
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
            

class GradientBoostingMSE:
    def __init__(
        self, n_estimators=100, learning_rate=0.1, max_depth=4, feature_subsample_size=None
    ):
        """
        n_estimators : int
            The number of trees in the forest.
        learning_rate : float
            //Unused
            Use alpha * learning_rate instead of alpha
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators: int = n_estimators
        self.learning_rate: float = learning_rate
        self.max_depth: int = max_depth
        self.feature_subsample_size = feature_subsample_size

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        y : numpy ndarray
            Array of size n_objects
        """
        s, q = X.shape
        self.estimators: list[DecisionTreeRegressor] = []
        self.alpha: list[int] = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, criterion='squared_error')
            self.estimators.append(tree)
        if self.feature_subsample_size is None:
            max_features = int(q / 3)
        else:
            max_features = self.feature_subsample_size
        seq: np.ndarray = np.arange(q).astype('int')
        self.list_indexes = []
        
        f = np.zeros((s))
        for estimator in self.estimators:
            shuffle(seq)
            indexes: np.ndarray = seq[:max_features]
            self.list_indexes.append(indexes.copy())
            estimator.fit(X[:, indexes], y - f)
            f += estimator.predict(X[:, indexes])

    def predict(self, X: np.ndarray):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        s, q = X.shape
        ans = np.zeros((s))
        
        for est, ind in zip(self.estimators, self.list_indexes):
            ans += est.predict(X[:, ind])
            
        return ans
    
    def get_params(self, deep=False):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'feature_subsample_size': self.feature_subsample_size
        }
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    
    