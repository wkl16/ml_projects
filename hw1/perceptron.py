import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class Perceptron:
    """Perceptron classifier.
    
    Parameters:
    eta : float - Learning rate (0.0 to 1.0)
    n_iter : int - Training epochs
    random_state : int - Seed for weight initialization
    
    Attributes:
    w_ : 1d-array - Weights after training
    b_ : float - Bias term
    errors_ : list - Misclassifications per epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Train the perceptron.
        
        Parameters:
        X : array-like, shape = [n_samples, n_features] - Training data
        y : array-like, shape = [n_samples] - Target labels

        Returns:
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Classify samples based on net input."""
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]  # Select two features (sepal length, petal length)
y = iris.target

mask = y != 2  # Exclude class 2 (Virginica)
X = X[mask]
y = y[mask]

y = np.where(y == 0, 0, 1)

# Scatter plot of dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Petal length (cm)')
plt.legend()
plt.title('Iris Dataset: Sepal vs. Petal Length')
plt.show()

# Train Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

# Plot training convergence
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title('Perceptron Training Convergence')
plt.show()
