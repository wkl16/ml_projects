import torch

class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier. Swapped numpy for pytorch.
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
    Weights after training.
    b_ : Scalar
    Bias unit after fitting.
    """

    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the
        number of examples and n_features is the
        number of features.
        y : array-like, shape = [n_examples]
        Target values.
        Returns
        -------
        self : Instance of LogisticRegressionGD
        """
        torch.manual_seed(self.random_state)
        self.w_ = torch.randn(X.shape[1], dtype=torch.float32) * 0.01
        self.b_ = torch.tensor(0.0, dtype=torch.float32)

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.matmul(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
        return self

    def net_input(self, X):
        """Calculate net input"""
        return X.matmul(self.w_) + self.b_

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return torch.sigmoid(z)

    def predict(self, X):
        """Return class label after unit step"""
        return (self.activation(self.net_input(X)) >= 0.5).int()
