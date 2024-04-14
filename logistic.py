import sys
import numpy as np
import matplotlib.pyplot as plt

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
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, decay_rate=0.99):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.decay_rate = decay_rate
        self.regularization_param = 0
        self.algorithm = None
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_predicted):
        return -np.mean(y * np.log(y_predicted + 1e-15) + (1 - y) * np.log(1 - y_predicted + 1e-15))
    
    def full_batch(self, X, y, is_regularized=False):
        num_samples, num_features = X.shape
        self.bias = 0
        self.algorithm = "full_batch"
        losses = []
        if is_regularized:
            self.regularization_param = self.calculate_regularization_param(X, y)
        self.weights = np.zeros(num_features)
        # Gradient descent
        for _ in range(self.max_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            current_loss = self.compute_loss(y, y_predicted)
            losses.append(current_loss)
            # Compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y)) + self.regularization_param * self.weights
            db = (1 / num_samples) * np.sum(y_predicted - y)
            # if is_regularized:
            #     print("logistic dw", dw)
            #     print("logistic db", db)
            # print(self.weights)
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < self.tolerance:
                break
        # print(len(losses))
        return losses
    
    def stochastic(self, X, y, is_regularized=False):
        _, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        self.algorithm = "stochastic"
        losses = []

        if is_regularized:
            self.regularization_param = self.calculate_regularization_param(X, y)
        self.weights = np.zeros(num_features)
        # Stochastic Gradient descent
        for _ in range(self.max_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            epoch_loss = self.compute_loss(y, y_predicted)
            losses.append(epoch_loss)
            for i in range(X.shape[0]):
                sample = X[i]
                label = y[i]
                linear_model = np.dot(sample, self.weights) + self.bias
                y_predicted = self.sigmoid(linear_model)
                # Compute gradients
                dw = sample * (y_predicted - label) + self.regularization_param * self.weights
                db = y_predicted - label
                # if is_regularized:
                #     print(dw)
                #     print(db)
                # print(self.weights)
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            self.learning_rate *= self.decay_rate
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < self.tolerance:
                break
        # print(len(losses))

        return losses

    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def calculate_regularization_param(self, X_train, y_train, num_folds=5):
        # Split data into folds
        _, num_features = X_train.shape
        fold_size = len(X_train) // num_folds
        folds_X = [X_train[i*fold_size:(i+1)*fold_size] for i in range(num_folds)]
        folds_y = [y_train[i*fold_size:(i+1)*fold_size] for i in range(num_folds)]
        
        # Store regularization parameter and its corresponding accuracy
        regularization_params = []
        accuracies = []
        
        for ln_lambda in range(0, -21, -5):
            lambda_val = np.exp(ln_lambda)
            accuracy_sum = 0
            # Cross-validation
            for i in range(num_folds):
                self.weights = np.zeros(num_features)

                X_val = folds_X[i]
                y_val = folds_y[i]
                X_tr = np.concatenate([folds_X[j] for j in range(num_folds) if j != i])
                y_tr = np.concatenate([folds_y[j] for j in range(num_folds) if j != i])
                if self.algorithm == "full_batch":
                    self.full_batch(X_tr, y_tr)
                else:
                    self.stochastic(X_tr, y_tr)
                y_pred = self.predict(X_val)
                accuracy_sum += np.mean(y_pred == y_val)
            
            regularization_params.append(lambda_val)
            accuracies.append(accuracy_sum / num_folds)
        
        best_param = regularization_params[np.argmax(accuracies)]
        print(f"Best regularization parameter: {best_param}, {np.log(best_param)}")
        return best_param

# Example usage:
if __name__ == "__main__":
    
    features, targets = read_arff("data/rice+cammeo+and+osmancik/Rice_Cammeo_Osmancik.arff")
    X_train, X_test, y_train, y_test = train_test_split(features, targets, random_state=42)
    model = LogisticRegression(learning_rate=0.01, max_iterations=100000)
    try:
        is_regularized = bool(sys.argv[1])
    except:
        is_regularized = False


    losses_gd = model.full_batch(X_train, y_train, is_regularized)
    losses_sgd = model.stochastic(X_train, y_train, is_regularized)
    plt.figure(figsize=(10, 5))
    plt.plot(losses_gd, label='Gradient Descent')
    plt.plot(losses_sgd, label='Stochastic Gradient Descent')
    plt.xlabel('Iteration (GD) / Epoch (SGD)')
    plt.ylabel('Loss')
    plt.title('Comparison of Loss Reduction')
    plt.legend()
    plt.show()
    
    # # Prediction
    # test_predictions = model.predict(X_test)

    # count = 0
    # correct = 0
    # for i in range(len(test_predictions)):
    #     if y_test[i] == test_predictions[i]:
    #         correct += 1
    #     count += 1
    
    # print(correct, count)

    # train_predictions = model.predict(X_train)
    # count = 0
    # correct = 0
    # for i in range(len(train_predictions)):
    #     if y_train[i] == train_predictions[i]:
    #         correct += 1
    #     count += 1
    
    # print(correct, count)