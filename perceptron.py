import numpy as np
import sys
import matplotlib.pyplot as plt


def train(data, labels):
    # weight = np.zeros(data.shape[1]) ## Zero vector initialization
    weight = np.random.rand(data.shape[1]) ## Random vector initialization
    print(f"Initial Weight: {weight}")
    converged = False
    num_of_iterations = 0
    while not converged:
        converged = True
        for i in range(data.shape[0]):
            x = data[i]
            y = labels[i]
            if (np.dot(x, weight) * y <= 0):
                weight += y * x
                converged = False
                num_of_iterations += 1
    return weight, num_of_iterations

def plot_data_and_boundaries(data, labels, weight, num_of_iterations, title):
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
    ax.set_title(f"{title}\nNumber of Iterations: {num_of_iterations}")
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
    weight, num_of_iterations = train(data, labels)
    print(f"Number of iterations: {num_of_iterations}")
    print(f"Weight: {weight}")
    plot_data_and_boundaries(data,
                             labels,
                             weight,
                             num_of_iterations,
                             f"Decision Boundary for {size.capitalize()} Dataset")

if __name__ == "__main__":
    main()
