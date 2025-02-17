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

# extract all features

X = df.iloc[:, [0,1,2,3]].values

# train perceptron 1 - setosa
ppn_s = Perceptron(eta=0.1, n_iter=10)
y_s = np.concatenate((np.ones(50, dtype=int), np.zeros(100, dtype=int)))
ppn_s.fit(X,y_s)

# train perceptron 2 - versicolor
ppn_ve = Perceptron(eta=0.1, n_iter=10)
y_ve = np.concatenate((np.zeros(50, dtype=int), np.ones(50, dtype=int), np.zeros(50, dtype=int)))
ppn_ve.fit(X,y_ve)

# train perceptron 3 - virginica
ppn_vi = Perceptron(eta=0.1, n_iter=10)
y_vi = np.concatenate((np.zeros(100, dtype=int), np.ones(50, dtype=int)))
ppn_vi.fit(X,y_vi)

# predict by choosing the class label with largest absolute net input value

test_data = [5.9, 3.0, 5.1, 1.8]

setosa_netinput = abs(ppn_s.net_input(test_data))
versicolor_netinput = abs(ppn_ve.net_input(test_data))
virginica_netinput = abs(ppn_vi.net_input(test_data))

print(f'setosa absolute net input value: {setosa_netinput}')
print(f'versicolor absolute net input value: {versicolor_netinput}')
print(f'virginica absolute net input value: {virginica_netinput}')

prediction = max(setosa_netinput, versicolor_netinput, virginica_netinput)

if prediction == setosa_netinput:
    print('Prediction is setosa')
elif prediction == versicolor_netinput:
    print('Prediction is versicolor')
else:
    print('Prediction is virginica')