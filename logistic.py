import sys
import numpy as np

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
                y.append(line[-1])
                x = []
                x.append(int(line[0]))
                x.append(map(float, line[1:5]))
                x.append(int(line[5]))
                x.append(float(line[6]))
                X.append(x)
                lines.append(line)
            else:
                break
    return np.array(X), np.array(y), lines



def read_arff(path):
    X = []
    y = []
    with open(path, 'r') as f:
        lines = f.readlines()
        data_start = False
        for line in lines:
            if not data_start:
                if line.strip().lower() == '@data':
                    data_start = True
            else:
                line = line.strip()
                line = line.replace("\n", "")
                line = line.split(",")
                if line[-1] == "Cammeo":
                    y.append(0)
                else:
                    y.append(1)
                x = []
                x.append(int(line[0]))
                x.extend(list(map(float, line[1:5])))
                x.append(int(line[5]))
                x.append(float(line[6]))
                X.append(x)
    
    X = np.array(X)
    y = np.array(y)
    
    # Scale features between 0 and 1
    X_scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    return X_scaled, y



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


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def full_batch(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            # print(self.weights)
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def stochastic(self, X, y):
        _, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        # Stochastic Gradient descent
        for _ in range(self.num_iterations):
            for i in range(X.shape[0]):
                sample = X[i]
                label = y[i]
                linear_model = np.dot(sample, self.weights) + self.bias
                y_predicted = self.sigmoid(linear_model)
                # Compute gradients
                dw = sample * (y_predicted - label)
                db = y_predicted - label
                # print(self.weights)
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
    
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

# Example usage:
if __name__ == "__main__":
    
    features, targets = read_arff("data/rice+cammeo+and+osmancik/Rice_Cammeo_Osmancik.arff")
    X_train, X_test, y_train, y_test = train_test_split(features, targets)
    model = LogisticRegression(learning_rate=0.01, num_iterations=10000)
    algorithm = sys.argv[1]
    if algorithm == "full_batch":
        model.full_batch(X_train, y_train)
    elif algorithm == "stochastic":
        model.stochastic(X_train, y_train)
    else:
        print("Invalid algorithm")
    
    # Prediction
    test_predictions = model.predict(X_test)

    count = 0
    correct = 0
    for i in range(len(test_predictions)):
        if y_test[i] == test_predictions[i]:
            correct += 1
        count += 1
    
    print(correct, count)

    train_predictions = model.predict(X_train)
    count = 0
    correct = 0
    for i in range(len(train_predictions)):
        if y_train[i] == train_predictions[i]:
            correct += 1
        count += 1
    
    print(correct, count)