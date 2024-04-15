import numpy as np
import logistic

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def read_data(path):
    lines = []
    X = []
    y = []
    with open(path, "r") as f:
        while(True):
            line = f.readline()
            if line:
                line = line.replace("\n", "")
                line = line.split(",")
                y.append(line[1])
                x = list(map(float, line[2:]))
                X.append(x)
                lines.append(line)
            else:
                break
    return np.array(X), np.array(y), lines

def read_data_for_logistic(path):
    X = []
    y = []
    with open(path, "r") as f:
        while(True):
            line = f.readline()
            if line:
                line = line.replace("\n", "")
                line = line.split(",")
                if line[1] == "B":
                    y.append(0)
                else:
                    y.append(1)
                x = list(map(float, line[2:]))
                X.append(x)
            else:
                break

    X = np.array(X)
    y = np.array(y)
    X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_scaled, y


class NaiveBayes:
    def __init__(self):
        self.class_probabilities = {}
        self.mean = {}
        self.variance = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.class_probabilities[c] = len(X_c) / n_samples

            self.mean[c] = np.mean(X_c, axis=0)
            self.variance[c] = np.var(X_c, axis=0)

    def calculate_gaussian(self, class_label, x):
        mean = self.mean[class_label]
        variance = self.variance[class_label]
        numerator = np.exp(-((x - mean) ** 2) / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = self.class_probabilities[c]
                likelihood = np.prod(self.calculate_gaussian(c, x))
                posterior = prior * likelihood
                posteriors[c] = posterior
            predictions.append(max(posteriors, key=posteriors.get))
        return predictions


if __name__ == "__main__":
    #targets: M = malignant B = benign
    #features: radius1, texture1, perimeter1, area1, smoothness1, compactness1, concavity1, concave_points1
    features, targets, lines = read_data("data/Breast Cancer Wisconsin Diagnostic/wdbc.data")
    X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=51)
    classifier = NaiveBayes()
    classifier.fit(X_train, y_train)

    train_predictions = classifier.predict(X_train)
    accuracy = np.mean(train_predictions == y_train)
    print("train accuracy:", accuracy)

    test_predictions = classifier.predict(X_test)
    accuracy = np.mean(test_predictions == y_test)
    print("test accuracy:", accuracy)

    features, targets = read_data_for_logistic("data/Breast Cancer Wisconsin Diagnostic/wdbc.data")
    X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=51)
    logistic_model = logistic.LogisticRegression(learning_rate=0.01, max_iterations=100000)
    logistic_model.full_batch(X_train, y_train)
    print("test for logistic regression")
    logistic.test_model(logistic_model, X_train, y_train, X_test, y_test)