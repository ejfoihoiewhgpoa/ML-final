import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k = 3):
        self.k = k

    def fit (self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, x):
        # Euclidean distance calculation
        distances = np.sqrt(((self.X_train - x)**2).sum(axis=1))
        # Get the indices of the k nearest neighbors
        index_points = np.argsort(distances)[:self.k]
        # Get the labels of the k nearest neighbors
        labels = self.y_train[index_points] 
        # Return the most common label among the k nearest neighbors
        return Counter(labels).most_common(1)[0][0]

    def predicts(self, X):
        return np.array([self.predict(x) for x in np.array(X)])

X_train = [[1, 2], [2, 3], [5, 5], [6, 5]]
y_train = [0, 0, 1, 1]
model = KNN(k = 3)
model.fit(X_train, y_train)
print(model.predicts([[1.5, 1.5], [5.5, 5]]))