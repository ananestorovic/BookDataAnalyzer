import numpy as np


class CustomLogisticRegression:
    def __init__(
        self,
        alpha=0.0001,
        max_iterations=1000,
        eta0=0.1,
        epsilon=0.0001,
        multi_class="ovr",
    ):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.eta0 = eta0
        self.epsilon = epsilon
        self.multi_class = multi_class
        self.weights = []
        self.intercepts = []
        self.classes = None

    def sigmoid(self, z):
        # Sigmoid funkcija se koristi za binarnu klasifikaciju
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-np.clip(z, -709, 709))),
            np.exp(np.clip(z, -709, 709))
            / (
                1 + np.exp(np.clip(z, -709, 709))
            ),
        )

    def softmax(self, z):
        # Softmax se koristi za multinomijalnu klasifikaciju
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        # Treniranje modela na osnovu ulaznih podataka (X) i target promenljive (y)
        X = np.array(X)
        y = np.array(y)
        self.classes = np.unique(y)

        # Izbor načina treniranja na osnovu vrste klasifikacije
        if self.multi_class == "ovr":
            self._fit_ovr(X, y)
        elif self.multi_class == "multinomial":
            self._fit_multinomial(X, y)

        return self

    def _update_weights(self, weights, intercept, error, x_i):
        weights -= self.eta0 * (error * x_i + self.alpha * weights)
        intercept -= self.eta0 * error
        return weights, intercept

    def _fit_ovr(self, X, y):
        # Treniranje modela za 'One-vs-Rest' klasifikaciju
        num_samples, num_features = X.shape

        for cls in self.classes:
            weights = np.zeros(num_features)
            intercept = 0
            y_binary = (y == cls).astype(int)

            for i in range(self.max_iterations):
                prev_weights = np.copy(weights)

                indices = np.random.permutation(num_samples) # Nasumično mešanje podataka
                X_shuffled = X[indices]
                y_shuffled = y_binary[indices]

                for i in range(num_samples):
                    x_i = X_shuffled[i]
                    y_i = y_shuffled[i]

                    z = np.dot(x_i, weights) + intercept
                    y_pred = self.sigmoid(z)

                    error = y_pred - y_i
                    weights, intercept = self._update_weights(
                        weights, intercept, error, x_i
                    )

                if np.linalg.norm(weights - prev_weights) < self.epsilon:
                    print(f"Stopped after {i} iterations due to minimal improvement.")
                    break

            self.weights.append(weights)
            self.intercepts.append(intercept)

    def _fit_multinomial(self, X, y):
        # Treniranje modela za multinomijalnu klasifikaciju
        num_samples, num_features = X.shape
        num_classes = len(self.classes)

        self.weights = np.zeros((num_classes, num_features))
        self.intercepts = np.zeros(num_classes)

        for i in range(self.max_iterations):
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(num_samples):
                x_i = X_shuffled[i]
                y_i = y_shuffled[i]

                scores = np.dot(self.weights, x_i) + self.intercepts
                probs = self.softmax(scores.reshape(1, -1)).ravel()

                # Ažuriranje težina za svaku klasu na osnovu greške
                for k in range(num_classes):
                    y_true = 1 if k == np.where(self.classes == y_i)[0][0] else 0
                    error = probs[k] - y_true
                    self.weights[k] -= self.eta0 * (error * x_i + self.alpha * self.weights[k])
                    self.intercepts[k] -= self.eta0 * error

    def predict(self, X):
        X = np.array(X)
        if self.multi_class == "ovr":
            return self._predict_ovr(X)
        else:  # multinomial
            return self._predict_multinomial(X)

    def _predict_ovr(self, X):
        num_samples = X.shape[0]
        num_classes = len(self.weights)
        class_scores = np.zeros((num_samples, num_classes))

        for j, (weights, intercept) in enumerate(zip(self.weights, self.intercepts)):
            class_scores[:, j] = np.dot(X, weights) + intercept

        return self.classes[np.argmax(class_scores, axis=1)]  # Izbor klase sa najvećim skorom

    def _predict_multinomial(self, X):
        scores = np.dot(X, self.weights.T) + self.intercepts
        return self.classes[np.argmax(scores, axis=1)]  # Izbor klase sa najvećim skorom

    def predict_proba(self, X):
        # Računanje verovatnoće za svaku klasu
        X = np.array(X)
        if self.multi_class == "ovr":
            return self._predict_proba_ovr(X)
        else:  # multinomial
            return self._predict_proba_multinomial(X)

    def _predict_proba_ovr(self, X):
        # Računanje verovatnoće za 'ovr' klasifikaciju
        num_samples = X.shape[0]
        num_classes = len(self.weights)
        proba = np.zeros((num_samples, num_classes))

        for j, (weights, intercept) in enumerate(zip(self.weights, self.intercepts)):
            proba[:, j] = self.sigmoid(np.dot(X, weights) + intercept)

        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def _predict_proba_multinomial(self, X):
        scores = np.dot(X, self.weights.T) + self.intercepts
        return self.softmax(scores)
