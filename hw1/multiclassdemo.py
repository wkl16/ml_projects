import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from perceptron import Perceptron

# load the Iris dataset into a DataFrame object

s = 'https://archive.ics.uci.edu/ml/'\
    'machine-learning-databases/iris/iris.data'

df = pd.read_csv(s, header=None, encoding='utf-8')

# extract string class labels and convert to integer class labels. 
# 0 - setosa, 1 - versicolor, 2 - virginica

y = df.iloc[:, 4].values
y = np.where(y == 'Iris-setosa', 0, y)
y = np.where(y == 'Iris-versicolor', 1, y)
y = np.where(y == 'Iris-virginica', 2, y)

print(y)
# extract all features

X = df.iloc[:, [0,1,2,3]].values

#print(X)

# TODO make sure to use perceptron class defined by Jay in part 2
# TODO figure out a better way of labeling positive instances

# train perceptron 1 - setosa
ppn_s = Perceptron(eta=0.1, n_iter=10)
y_s = np.where(y == 2, 1, y)
print(y_s)

ppn_s.fit(X,y_s)

# train perceptron 2 - versicolor
ppn_ve = Perceptron(eta=0.1, n_iter=10)
y_ve = np.where(y == 2, 0, y)
print(y_ve)

ppn_ve.fit(X,y_ve)

# train perceptron 3 - virginica
ppn_vi = Perceptron(eta=0.1, n_iter=10)
#y_vi = np.where(y == 2, 1, y)
y_vi = np.where(y == 1, 0, y)
print(y_vi)

ppn_vi.fit(X,y_vi)