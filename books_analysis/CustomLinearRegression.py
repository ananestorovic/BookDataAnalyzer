import numpy as np


class CustomLinearRegression:
    def __init__(self, alpha=0.01, iterations=300, threshold=1e-4):
        self.alpha = alpha
        self.iterations = iterations
        self.threshold = threshold
        self.W = None
        self.w0 = None

    def fit(self, X, y):
        samples, features = X.shape

        self.W = np.zeros((features, 1))
        self.w0 = 0
        prev_cost = np.inf

        # Stohasticki gradijentni spust
        for iteration in range(self.iterations):
            pred_y = self.w0 + np.dot(X, self.W)

            # Azuriranje tezina
            self.w0 -= self.alpha * (1 / samples) * np.sum(pred_y - y)
            self.W -= self.alpha * (1 / samples) * np.dot(X.T, (pred_y - y))


            # Funkcija greske
            cur_cost = np.mean((y - pred_y) ** 2)

            # Provera uslova za zaustavljanje
            if abs(cur_cost - prev_cost) < self.threshold:
                print(f"Stopped at iter {iteration + 1}, cost diff {abs(cur_cost - prev_cost)} < {self.threshold}")
                break

            prev_cost = cur_cost
            print('MOJ ALGORITAM!!!!!!!')

    # Funkcija za predikciju
    def predict(self, X):
        return self.w0 + np.dot(X, self.W)