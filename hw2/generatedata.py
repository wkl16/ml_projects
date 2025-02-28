# Task 2
# Ellie Larence

import numpy as np
import matplotlib.pyplot as plt

def make_classification(d, n, u = 1, random_seed = 2):
    """ Generate linearly separable data based on a random separation hyperplane.

    Parameters
    ----------
    d : int. dimension of data
    n : int. number of data points
    u : int. range from which data points will be generated. Will form a symmetric bounding box. default 1.
    random_seed : int. random number generator seed. default 2.

    Returns
    -------
    A set of linearly separable data labeled either -1 or 1 based on which side of the hyperplane they fall.  
    samples : numpy array where each row is a d-dimensional samples
    labels : a 1D numpy array of length n consisting of -1 or 1

    """
    # set seed for reproducibility
    np.random.seed(random_seed)

    # step 1 - randomly generate a d-dimensional vector a
    a = np.random.rand(d)

    # step 2 - randomly select n samples in the range of [-u, u] in each dimension. We will use a uniform distribution
    samples = np.random.uniform(-u, u, (n, d))
    
    # step 3 - give each xi a label yi such that if aTx < 0 then yi = -1, otherwise yi = 1
    labels = np.where(np.dot(samples, a) < 0, -1, 1)

    # Below is optional, comment out if you don't want to create a plot
    # plot samples and hyperplane - only tested for 2-D data and u = 1

    plt.figure(figsize=(8,6))
    colors = ['red' if label == -1 else 'blue' for label in labels]
    plt.scatter(samples[:, 0], samples[:, 1], c=colors)
    a_x = np.linspace(-u, u, 100)
    a_y = -(a[0] / a[1]) * a_x
    plt.ylim(-1.5,1.5)
    plt.xlim(-1.5,1.5)
    plt.plot(a_x, a_y, 'k--', label = 'Separating Hyperplane - vector a')
    plt.axhline(y=-u, color='green', linestyle='--', label='Bounding Box - [-u, u]')
    plt.axhline(y=u, color='green', linestyle='--')
    plt.axvline(x=-u, color='green', linestyle='--')
    plt.axvline(x=u, color='green', linestyle='--')
    plt.title ('Linearly Separable Data')
    plt.legend()
    # plt.show()

    return samples, labels

def data_to_txt(samples, labels, filename):
    data = np.hstack((samples, labels.reshape(-1,1)))
    np.savetxt(filename, data, delimiter=',')

dimensions = [10, 50, 100, 500, 1000]
# sample_counts = [500, 1000, 5000, 10000, 1000000]
sample_counts = [10000, 1000000]

for d in dimensions:
    for s in sample_counts:
        samples, labels = make_classification(d, s)
        filename = f"samples_{d}d_{s}s.txt" 
        data_to_txt(samples, labels, filename)

# for d in dimensions:
#     for n in sample_counts:
#         samples, labels = make_classification(d, n)
#         filename = f"samples_{d}d_{n}s.txt" 
#         data_to_txt(samples, labels, filename)