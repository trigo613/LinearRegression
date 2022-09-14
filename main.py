import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston


class Linear_Regression:
    def __init__(self):
        self.x = False #Weights
        self.b = 1  #Bias
        self.lr = 0.01  #Learning rate

    def predict(self, X):
        prediction = np.dot(X, self.x) + self.b
        return prediction

    # X input must be a numpy matrix, the y vector must be a column vector

    def train(self, X, y, epochs):
        # Creating weight matrix if it hasn't been created yet
        if type(self.x) == bool:
            self.x = np.random.random((X.shape[1], 1))

        for epoch in range(epochs):
            prediction = self.predict(X)
            mean_factor = (2 / X.shape[0])
            gradient = mean_factor * (np.dot(X.transpose(), prediction - y))
            self.x = self.x - self.lr * gradient
            self.b = self.b - self.lr * np.mean((prediction - y))


# Loading Data
data = load_boston()
X = data['data']
y = data['target'].reshape(506, 1)

# Scaling data
ms = MinMaxScaler()
X = ms.fit_transform(X)

lr = Linear_Regression()

lr.train(X, y, 20000)

pred = lr.predict(X)
print(mean_absolute_error(pred, y))