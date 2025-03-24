import numpy as np
from regressionTree import RegressionTree

if __name__ == "__main__":
    # Test
    #X = np.array([[1, 2], [8, 3], [3, 4], [4, 5], [5, 6]])
    #y = np.array([1, 2, 3, 4, 5])
    #X = np.array([[2, 3], [1, 4], [3, 5], [5, 1], [4, 2], [6, 3], [7, 7]])
    #y = np.array([7, 6, 8, 3, 5, 7, 9])
    #test = np.array([[3, 4], [6, 2]])
    X = np.array([[2, 3, 1, 4], [1, 4, 2, 5], [3, 5, 3, 6], [5, 1, 4, 2], [4, 2, 5, 3],
                  [6, 3, 6, 4], [7, 7, 7, 5], [8, 6, 8, 6], [9, 5, 9, 7], [10, 4, 10, 8]])

    y = np.array([7, 6, 8, 3, 5, 7, 9, 8, 10, 12])
    test = np.array([[2, 3, 1, 4], [1, 3, 8, 10], [8, 8, 2, 6]])
    # Initialize and train the regression tree
    #reg_tree = RegressionTree(X=X, y=y)
    #reg_tree = RegressionTree(X=X, y=y, limit='leaf_size', leaf_size=5)
    reg_tree = RegressionTree(X=X, y=y, limit='height', max_height=2)
    reg_tree.fit()
    print("Regression Tree: ")
    reg_tree.print_tree()
    print("")
    predictions = reg_tree.predict(test)
    print("")
    print("Predictions: ", predictions)
    print("")
    print("Decision Path:")
    reg_tree.decision_path(test)
