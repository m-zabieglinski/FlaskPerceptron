import numpy as np
import pickle
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(data = np.c_[iris["data"], iris["target"]], columns = iris["feature_names"]+["target"])

df.drop(df.index[df["target"] == 2], inplace = True)
X = df.loc[:, ["petal length (cm)", "sepal length (cm)"]].values
y = df.loc[:, ["target"]].values


class Perceptron():
    def __init__(self, eta = 0.01, n = 20):
        self.eta = eta
        self.n = n
    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for k in range(self.n):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)
    
model = Perceptron()
model.fit(X, y)

with open("perc_iris.pkl", "wb") as this_model:
    pickle.dump(model, this_model)
