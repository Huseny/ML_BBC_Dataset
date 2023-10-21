import numpy as np


class LogisticRegression:
    def __init__(self) -> None:
        pass

    def learn(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        learning_rate: float,
        num_iter: int,
    ) -> list:
        self.train_x = train_x
        self.train_y = train_y
        self.weights = []

        X = np.column_stack((np.ones(len(self.train_x)), self.train_x))
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        num_classes = len(np.unique(self.train_y))

        for i in range(num_classes):
            binary_y = np.where(self.train_y == i, 1, 0)

            theta = np.zeros(X.shape[1])

            for _ in range(num_iter):
                z = np.dot(X, theta)
                h = self.sigmoid(z)

                error = h - binary_y
                gradient = np.dot(X.T, error) / len(X)

                theta -= learning_rate * gradient

            self.weights.append(theta)
        return self.weights

    def classify(self, data: np.ndarray, weights: list):
        data = np.hstack((np.ones((data.shape[0], 1)), data))
        test = np.hstack((np.ones((data.shape[0], 1)), data))

        scores = np.dot(test, np.transpose(weights))

        return np.argmax(scores, axis=1)

    def sigmoid(self, z) -> float:
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
