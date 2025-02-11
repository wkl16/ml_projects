import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap
class AdalineGD:
    """Adaptive Linear Neuron Classifier (Adaline)
    
    Parameters:
    eta : float - Learning rate (0.0 to 1.0)
    n_iter : int - Training epochs
    random_state : int - Seed for weight initialization
    
    Attributes:
    w_ : 1D array - Weights after training
    b_ : float - Bias term
    losses_ : list - Mean squared error per epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Train Adaline using batch gradient descent.
        
        Parameters:
        X : array-like, shape = [n_samples, n_features] - Training data
        y : array-like, shape = [n_samples] - Target labels

        Returns:
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            self.losses_.append((errors ** 2).mean())  # MSE loss
        return self

    def net_input(self, X):
        """Compute linear combination of inputs and weights."""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Identity activation function"""
        return X

    def predict(self, X):
        """Return class label after unit step function."""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    def hyperplane(self, X, y, title="Decision Regions", resolution=0.01):
        """Plot decision regions using a meshgrid."""
        colors = ('red', 'blue')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        x0_range = np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, resolution)
        x1_range = np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, resolution)
        grid_x0, grid_x1 = np.meshgrid(x0_range, x1_range)
        grid_points = np.c_[grid_x0.ravel(), grid_x1.ravel()]
        Z = self.predict(grid_points).reshape(grid_x0.shape)
        
        plt.figure(figsize=(8, 6))
        plt.contourf(grid_x0, grid_x1, Z, alpha=0.3, cmap=cmap)
        plt.xlim(grid_x0.min(), grid_x0.max())
        plt.ylim(grid_x1.min(), grid_x1.max())

        plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='o', label='Setosa')
        plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='s', label='Versicolor')
        plt.xlabel('Sepal length (cm)')
        plt.ylabel('Petal length (cm)')
        plt.title(title)
        plt.legend()
        plt.show()
    
if __name__ == '__main__': 
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]  # Use Sepal Length & Petal Length
    y = iris.target

    # keep only Setosa (0) and Versicolor (1), removing Virginica (2)
    mask = y != 2
    X, y = X[mask], y[mask]

    # convert labels: Setosa → 0, Versicolor → 1
    y = np.where(y == 0, 0, 1)

    adaline = AdalineGD(eta=0.01, n_iter=20)
    adaline.fit(X, y)
    adaline.hyperplane(X, y, title="Adaline on Iris Dataset")

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(adaline.losses_) + 1), adaline.losses_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Adaline Training Loss')
    plt.show()
