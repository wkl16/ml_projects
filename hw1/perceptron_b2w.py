#Task 2
#Modified by Jamerson Tenorio
import numpy as np

class PerceptronBiasAsWeight:
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    errors_ : list
    Number of misclassifications (updates) in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of
        examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
        Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)

        # Appending 1 to X's for the bias to be converted as a weight.
        b2w = list()
        for row in X:
            b2w.append([1.0] + list(row))

        # Need to update new array to be a numpy array.
        b2w = np.array(b2w)

        # Update self.w_ to use updated numpy array.
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=b2w.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            # Update loop to zip on b2w array instead of original X array.
            for xi, target in zip(b2w, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)