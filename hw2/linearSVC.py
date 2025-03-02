import numpy as np

class LinearSVC:
    """Linear SVC classifier (Primal L2 Regularization)
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.
    C : float (C > 0)
    Regularization parameter.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    b_ : Scalar
    Bias unit after fitting.
    losses_ : list
    Hinge loss function with L2 regularization in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1, C=1.0):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.C = C

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
        # Initialize weight and bias
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []
        n = len(y)

        for _ in range(self.n_iter):
            # In each epoch check hinge loss case for each sample
            for i in range(n):
                #Hinge loss case for ith sample
                case = 1 - (y[i] * (np.dot(X[i], self.w_) + self.b_))

                # Case 1
                # Correct classification and not marginal
                # Do not need to penalize, only update w
                if case <= 0:
                    self.w_ = self.w_ - (self.eta * self.C * self.w_)

                # Case 2
                # May or may not be misclassified (can be "marginally" correct)
                # will still need to penalize to make more robust and correct.
                # Need to update weight and bias
                elif case > 0:
                    self.w_ -= self.eta * ( (1 / n) * (self.w_ - self.C * y[i] * X[i]) )
                    self.b_ = self.b_ - (self.eta * (self.C / n) * (-y[i]))
            loss = self.hinge_loss_L2(X, y, self.C, self.w_)
            self.losses_.append(loss)
        return  self

    def hinge_loss_L2(self, X, y, C, w):
        """Hinge loss calculation for primal L2 regularization"""
        n = len(y)
        return ((C/n) * np.sum(np.maximum(0, 1 - y * self.net_input(X)))) + ((1/2) * (np.linalg.norm(w)) ** 2)

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

