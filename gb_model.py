import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")




class DecisionTreeRegressorFromScratch:

    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.tree_ = {}
        self.max_depth_ = max_depth

    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def fit(self, X, y, tree_path='0'):
        if len(tree_path) - 1 == self.max_depth_ or X.shape[0] <= 1:
            self.tree_[tree_path] = np.mean(y)
            return

        minimum_mse = None
        best_split = None

        for feature in range(X.shape[1]):
            for value in sorted(set(X[:, feature])):
                less_than_or_equal_obs = X[:, feature] <= value
                X1, y1 = X[less_than_or_equal_obs], y[less_than_or_equal_obs]
                X2, y2 = X[~less_than_or_equal_obs], y[~less_than_or_equal_obs]

                MSE1 = self.mse(y1, np.mean(y1))
                MSE2 = self.mse(y2, np.mean(y2))
                weight_1 = len(y1) / len(y)
                weight_2 = len(y2) / len(y)
                weighted_mse = MSE1 * weight_1 + MSE2 * weight_2

                if minimum_mse is None or weighted_mse < minimum_mse:
                    minimum_mse = weighted_mse
                    best_split = (feature, value)

        feature, value = best_split
        splitting_condition = X[:, feature] <= value
        X1, y1 = X[splitting_condition], y[splitting_condition]
        X2, y2 = X[~splitting_condition], y[~splitting_condition]

        self.tree_[tree_path] = best_split
        self.fit(X1, y1, tree_path=tree_path + '0')
        self.fit(X2, y2, tree_path=tree_path + '1')

    def predict(self, X):
        results = []
        for i in range(X.shape[0]):
            tree_path = '0'
            while True:
                value_for_path = self.tree_[tree_path]
                if type(value_for_path) != tuple:
                    result = value_for_path
                    break
                feature, value = value_for_path
                if X[i, feature] <= value:
                    tree_path += '0'
                else:
                    tree_path += '1'
            results.append(result)
        return np.array(results)

class GradientBoostingRegressorFromScratch:
    def __init__(self, tree_levels, number_trees, learning_rate, subsample=1.0, max_features=None):
        self.tree_levels = tree_levels
        self.number_trees = number_trees
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        residuals = y - np.mean(y)
        initial_prediction = np.mean(y)
        self.initial_prediction = initial_prediction

        for i in range(self.number_trees):
            if self.subsample < 1.0:
                subsample_indices = np.random.choice(len(X), int(len(X) * self.subsample), replace=False)
                X_subsample, y_subsample = X[subsample_indices], residuals[subsample_indices]
            else:
                X_subsample, y_subsample = X, residuals

            if self.max_features:
                feature_indices = np.random.choice(X.shape[1], self.max_features, replace=False)
                X_subsample = X_subsample[:, feature_indices]
            else:
                feature_indices = None

            tree = DecisionTreeRegressorFromScratch(self.tree_levels)
            tree.fit(X_subsample, y_subsample)

            if feature_indices is not None:
                tree.feature_indices = feature_indices
            else:
                tree.feature_indices = np.arange(X.shape[1])

            self.trees.append(tree)
            residuals -= self.learning_rate * tree.predict(X)

    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            if hasattr(tree, 'feature_indices'):
                X_subset = X[:, tree.feature_indices]
            else:
                X_subset = X
            predictions += self.learning_rate * tree.predict(X_subset)
        return predictions
    


def open_csv(name):
    df = pd.read_csv(f'Data/{name}.csv')
    X=df[['Open', 'High', 'Low', 'Adj Close', 'Volume']].values
    y=df['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test

def open_X(name):
    df = pd.read_csv(f'Data/{name}.csv')
    X=df[['Open', 'High', 'Low', 'Adj Close', 'Volume']].values
    return X


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

#gb = GradientBoostingRegressorFromScratch(tree_levels=4, number_trees=500, learning_rate=0.05)
#gb.fit(X_train, y_train)
#y_pred = gb.predict(X_test)


