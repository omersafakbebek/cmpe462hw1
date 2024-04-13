import numpy as np
import sys
import matplotlib.pyplot as plt

def seperate_data(data, labels, ratio):
    N = data.shape[0]
    indices = np.random.permutation(N)
    data = data[indices]
    labels = labels[indices]

    N_train = int(ratio * N)
    train_data = data[:N_train]
    train_labels = labels[:N_train]
    test_data = data[N_train:]
    test_labels = labels[N_train:]
    return train_data, train_labels, test_data, test_labels

def train(data, labels):
    weight = np.zeros(data.shape[1])
    converged = False
    epoch = 0
    while not converged:
        converged = True
        for i in range(data.shape[0]):
            x = data[i]
            y = labels[i]
            if (np.dot(x, weight) * y <= 0):
                weight += y * x
                converged = False
        epoch += 1
    return weight

def test(data, labels, weight):
    correct = 0
    for i in range(data.shape[0]):
        x = data[i]
        y = labels[i]
        if (np.dot(x, weight) * y > 0):
            correct += 1
    accuracy = correct / data.shape[0]
    return accuracy

def plot_data_and_boundaries(data, labels, weight, title):
    _, ax = plt.subplots()
    data = data[:, -2:]  
    weight_relevant = weight[-2:] 

    for label in np.unique(labels):
        idx = np.where(labels == label)
        ax.scatter(data[idx, 0], data[idx, 1], label=f"Class {label}", s=10)
    x_values = np.array([np.min(data[:, 0]), np.max(data[:, 0])])
    y_values = -(weight_relevant[0] * x_values + weight[0]) / weight_relevant[1] 
    ax.plot(x_values, y_values, 'k--', label='Decision Boundary')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    plt.show()

def main():
    size = sys.argv[1]
    if size not in ["small", "large"]:
        print("Invalid size. Please enter 'small' or 'large'.")
        return
    data_file = f"data/data_{size}.npy"
    label_file = f"data/label_{size}.npy"

    data = np.load(data_file)
    labels = np.load(label_file)

    if size == "large":
        ratio = 0.8
    else:
        ratio = 0.9
    train_data, train_labels, test_data, test_labels = seperate_data(data, labels, ratio)
    weight = train(train_data, train_labels)
    accuracy = test(test_data, test_labels, weight)
    print(f"Accuracy: {accuracy}")

    plot_data_and_boundaries(data,
                             labels,
                             weight,
                             f"Decision Boundary for {size.capitalize()} Dataset")

if __name__ == "__main__":
    main()
