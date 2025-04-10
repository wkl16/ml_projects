import numpy as np

from generate_data import generate_data_numbers
from generate_data import generate_data_fashion
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def numbers_PCA():
    X_train, X_test, y_train, y_test = generate_data_numbers()

    print("Creating PCA's for Numbers Dataset")
    #Training Dataset
    dimensions = [50, 100, 200]
    pca50_train = PCA(n_components=dimensions[0])
    pca100_train = PCA(n_components=dimensions[1])
    pca200_train = PCA(n_components=dimensions[2])
    pca50_train.fit_transform(X_train)
    pca100_train.fit_transform(X_train)
    pca200_train.fit_transform(X_train)

    #Testing Dataset
    '''
    pca50_test = PCA(n_components=dimensions[0])
    pca100_test = PCA(n_components=dimensions[1])
    pca200_test = PCA(n_components=dimensions[2])
    pca50_test.fit_transform(X_test)
    pca100_test.fit_transform(X_test)
    pca200_test.fit_transform(X_test)
    '''
    # Info for PCA's
    variance_ratio = list()
    cumulative_variance = list()
    variance_ratio.append(pca50_train.explained_variance_ratio_)
    variance_ratio.append(pca100_train.explained_variance_ratio_)
    variance_ratio.append(pca200_train.explained_variance_ratio_)
    cumulative_variance.append(np.cumsum(pca50_train.explained_variance_ratio_))
    cumulative_variance.append(np.cumsum(pca100_train.explained_variance_ratio_))
    cumulative_variance.append(np.cumsum(pca200_train.explained_variance_ratio_))

    print("Best individual principal components:", cumulative_variance[0][0])
    print("Cumulative variance of principal components at 50:", cumulative_variance[0][-1])
    print("Cumulative variance of principal components at 100:", cumulative_variance[1][-1])
    print("Cumulative variance of principal components at 200:", cumulative_variance[2][-1])

    fig, axs = plt.subplots(3, 1)
    j = .8
    k = 0.55
    for i, n in enumerate(dimensions):
        #axs[i].plot(range(1, n+1), cumulative_variance[i], marker='o')
        axs[i].bar(range(1, n+1), variance_ratio[i], align='center', label='Individual explained variance')
        axs[i].step(range(1, n+1), cumulative_variance[i], where='mid', label='Cumulative explained variance')
        axs[i].text(variance_ratio[i][0]+j, variance_ratio[i][0], f'{variance_ratio[i][0]:.2f}', ha='center', va='bottom', fontsize=10, color='blue')
        axs[i].text(n+k, cumulative_variance[i][-1]-.06, f'{cumulative_variance[i][-1]:.2f}', ha='center', va='bottom',fontsize=10, color='green')
        axs[i].set_title(f'Total Variance Captured by {n} Principal Components for Numbers Dataset')
        axs[i].set_xlabel('Principal components index')
        axs[i].set_ylabel('Explained variance ratio')
        axs[i].legend(loc='lower right')
        axs[i].grid(True)
        j -= .8
        k += 1

    plt.tight_layout()
    plt.show()

def fashion_PCA():
    X_train, X_test, y_train, y_test = generate_data_fashion()

    print("Creating PCA's for Fashion Dataset")
    #Training Dataset
    dimensions = [50, 100, 200]
    pca50_train = PCA(n_components=dimensions[0])
    pca100_train = PCA(n_components=dimensions[1])
    pca200_train = PCA(n_components=dimensions[2])
    pca50_train.fit_transform(X_train)
    pca100_train.fit_transform(X_train)
    pca200_train.fit_transform(X_train)

    #Testing Dataset
    '''
    pca50_test = PCA(n_components=dimensions[0])
    pca100_test = PCA(n_components=dimensions[1])
    pca200_test = PCA(n_components=dimensions[2])
    pca50_test.fit_transform(X_test)
    pca100_test.fit_transform(X_test)
    pca200_test.fit_transform(X_test)
    '''
    # Info for PCA's
    variance_ratio = list()
    cumulative_variance = list()
    variance_ratio.append(pca50_train.explained_variance_ratio_)
    variance_ratio.append(pca100_train.explained_variance_ratio_)
    variance_ratio.append(pca200_train.explained_variance_ratio_)
    cumulative_variance.append(np.cumsum(pca50_train.explained_variance_ratio_))
    cumulative_variance.append(np.cumsum(pca100_train.explained_variance_ratio_))
    cumulative_variance.append(np.cumsum(pca200_train.explained_variance_ratio_))

    print("Best individual principal components:", cumulative_variance[0][0])
    print("Cumulative variance of principal components at 50:", cumulative_variance[0][-1])
    print("Cumulative variance of principal components at 100:", cumulative_variance[1][-1])
    print("Cumulative variance of principal components at 200:", cumulative_variance[2][-1])

    fig, axs = plt.subplots(3, 1)
    j = .8
    k = 0.55
    for i, n in enumerate(dimensions):
        #axs[i].plot(range(1, n+1), cumulative_variance[i], marker='o')
        axs[i].bar(range(1, n+1), variance_ratio[i], align='center', label='Individual explained variance')
        axs[i].step(range(1, n+1), cumulative_variance[i], where='mid', label='Cumulative explained variance')
        axs[i].text(variance_ratio[i][0]+j, variance_ratio[i][0], f'{variance_ratio[i][0]:.2f}', ha='center', va='bottom', fontsize=10, color='blue')
        axs[i].text(n+k, cumulative_variance[i][-1]-.06, f'{cumulative_variance[i][-1]:.2f}', ha='center', va='bottom',fontsize=10, color='green')
        axs[i].set_title(f'Total Variance Captured by {n} Principal Components for Fashion Dataset')
        axs[i].set_xlabel('Principal components index')
        axs[i].set_ylabel('Explained variance ratio')
        axs[i].legend(loc='lower right')
        axs[i].grid(True)
        j -= .8
        k += 1

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    numbers_PCA()
    fashion_PCA()
