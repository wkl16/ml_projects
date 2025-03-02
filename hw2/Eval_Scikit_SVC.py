from sklearn import datasets
from sklearn.model_selection import train_test_split 
import numpy as np

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

print(iris)

# print('Class labels:', np.unique(y))

X_train, X_test,  y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state=1, stratify=y
    )

# print('Labels counts in y:', np.bincount(y))
# print('Labels counts in y_train:', np.bincount(y_train))
# print('Labels counts in y_test:', np.bincount(y_test))

#2.422676655385898048e-01,
# 5.828418855407813126e-02,
# -7.308401093101328794e-01,
# 2.715624253149284684e-02,
# -6.311202687061694405e-01,
# 5.706702956333469245e-01,
# 7.079505852789775844e-01,
# -1.152632523614438576e-02,
# 6.931229707149360042e-01,
# -8.407090459818780115e-01,
# -1.000000000000000000e+00