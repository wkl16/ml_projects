import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Adaline_T4 import AdalineSGD
from Adaline_T4 import AdalineGD
from DecisionRegions import plot_decision_regions

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
d = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
print('From URL:', s)
df = pd.read_csv(s,header=None,encoding='utf-8')
df.tail()

# select setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
# plot data
plt.scatter(X[:50, 0], X[:50, 1],
color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
color='blue', marker='s', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada_gd = AdalineGD(n_iter=20, eta=0.02, random_state=10)
ada_gd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_gd.losses_) + 1),ada_gd.losses_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean squared error')
plt.tight_layout()
plt.show()

ada_sgd = AdalineSGD(n_iter=20, eta=0.02, random_state=10)
ada_sgd.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('Adaline - Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_,
marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.show()

ada_mb_sgd = AdalineSGD(n_iter=20, eta=0.02, random_state=10)
ada_mb_sgd.fit_mini_batch_SGD(X_std, y)
plot_decision_regions(X_std, y, classifier=ada_mb_sgd)
plt.title('Adaline - Mini Batch Stochastic gradient descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('Petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1, len(ada_mb_sgd.losses_) + 1), ada_mb_sgd.losses_,
marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average loss')
plt.tight_layout()
plt.show()