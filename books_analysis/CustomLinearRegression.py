import numpy as np


class CustomLinearRegression:
    def __init__(
        self, alpha: float = 0.005, iterations: int = 800000, threshold: float = 1e-5
    ):
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

        treshold_counter = 0

        # Stohasticki gradijentni spust
        for iteration in range(self.iterations):
            pred_y = self.w0 + np.dot(X, self.W)

            # Azuriranje tezina
            self.w0 -= self.alpha * (1 / samples) * np.sum(pred_y - y)
            self.W -= self.alpha * (1 / samples) * np.dot(X.T, (pred_y - y))

            # Funkcija greske
            cur_cost = np.mean((y - pred_y) ** 2)

            # Provera uslova za zaustavljanje
            if (prev_cost - cur_cost) < self.threshold:
                print(
                    f"Stopped at iter {iteration + 1}, cost diff {abs(cur_cost - prev_cost)} < {self.threshold}"
                )
                treshold_counter += 1
                if treshold_counter > 5:
                    break
                # break

            prev_cost = cur_cost

    # Funkcija za predikciju
    def predict(self, X):
        return self.w0 + np.dot(X, self.W)
