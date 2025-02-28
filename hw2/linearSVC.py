import numpy as np

class LinearSVC:

    def __init__(self, eta=0.01, n_iter=50, random_state=1, C=1.0):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.C = C

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []
        n = len(y)
        for _ in range(self.n_iter):
            for i in range(n):
                case = 1 - (y[i] * (np.dot(X[i], self.w_) + self.b_))
                if case <= 0:
                    #print("case 1")
                    #not marginal, do not need to penalize, only update w
                    self.w_ = self.w_ - (self.eta * self.C * self.w_)
                elif case > 0:
                    #print("case 2")
                    #need to penalize, update weight and bias
                    # self.w_ = self.w_ - (self.eta * (self.C * self.w_ - y[i] * X[i]))
                    # self.b_ = self.b_ - (self.eta * (-y[i]))
                    self.w_ -= self.eta * ( (1 / n) * (self.w_ - self.C * y[i] * X[i]) )
                    self.b_ = self.b_ - (self.eta * (self.C / n) * (-y[i]))
            loss = self.hinge_loss_L2(X, y, self.C, self.w_)
            self.losses_.append(loss)
        return  self

    def hinge_loss_L2(self, X, y, C, w):
        n = len(y)
        return ((C/n) * np.sum(np.maximum(0, 1 - y * self.net_input(X)))) + ((1/2) * (np.linalg.norm(w)) ** 2)

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


