import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


name = "AMD"
df = pd.read_csv('Data/AMD.csv')
df.head()

X=df[['Open', 'High', 'Low', 'Close', 'Volume']].values
y=df['Adj Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        # Calculate the variance of the current dataset
        best_var = np.var(y) * m
        best_idx, best_thr = None, None

        for idx in range(n):
            thresholds, variances = zip(*sorted(zip(X[:, idx], y)))
            for i in range(1, m):
                left = variances[:i]
                right = variances[i:]
                var_left = np.var(left) * len(left)
                var_right = np.var(right) * len(right)
                var_total = var_left + var_right

                if thresholds[i] == thresholds[i - 1]:
                    continue
                if var_total < best_var:
                    best_var = var_total
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        node = {
            'value': np.mean(y)
        }

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['feature_index'] = idx
                node['threshold'] = thr
                node['left'] = self._build_tree(X_left, y_left, depth + 1)
                node['right'] = self._build_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree
        while 'threshold' in node:
            if inputs[node['feature_index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['value']


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        for _ in range(self.n_estimators):
            # Bootstrap Aggregation (Bagging)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # Average predictions
        return np.mean(tree_preds, axis=0)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


rf = RandomForestRegressor(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
print(mse(y_test, predictions))