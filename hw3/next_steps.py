import numpy as np

from generate_data import generate_data_numbers
from generate_data import generate_data_fashion
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data_numbers()

    # Assuming this is the next step of the assignment here
    #A3_pipe = make_pipeline(StandardScaler(), SVC(random_state=42))
    #print(A3_pipe.fit(X_train, y_train).score(X_test, y_test))
    
    #3.2 Dimensionality reduction
    print("Creating PCA's")
    dimensions = [50, 100, 200]
    pca50 = PCA(n_components=dimensions[0])
    pca100 = PCA(n_components=dimensions[1])
    pca200 = PCA(n_components=dimensions[2])
    X_pca50 = pca50.fit_transform(X_train)
    X_pca100 = pca100.fit_transform(X_train)
    X_pca200 = pca200.fit_transform(X_train)

    variance_ratio = list()
    cumulative_variance = list()
    variance_ratio.append(pca50.explained_variance_ratio_)
    variance_ratio.append(pca100.explained_variance_ratio_)
    variance_ratio.append(pca200.explained_variance_ratio_)
    cumulative_variance.append(np.cumsum(pca50.explained_variance_ratio_))
    cumulative_variance.append(np.cumsum(pca100.explained_variance_ratio_))
    cumulative_variance.append(np.cumsum(pca200.explained_variance_ratio_))

    print("Cumulative variance of dimension reduced to 50:", cumulative_variance[0][-1])
    print("Cumulative variance of dimension reduced to 100:", cumulative_variance[1][-1])
    print("Cumulative variance of dimension reduced to 200:", cumulative_variance[2][-1])

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    for i, n in enumerate(dimensions):
        axs[i].plot(range(1, n+1), cumulative_variance[i], marker='o')
        axs[i].set_title(f'Cumulative Variance for {n} Principal Components')
        axs[i].set_xlabel('Number of Principal Components')
        axs[i].set_ylabel('Cumulative Variance')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()
