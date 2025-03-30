import numpy as np

class RegressionTree:
    """Regression Tree Classifier (Sum of Squared Error)
    Parameters
    -----------
    X : {array-like}, shape = [num_samples, num_features]
    Training vectors, where n_examples is the number of
    examples and n_features is the number of features.
    y : {array-like}, shape = [num_samples]
    Target values.
    max_height : float
    If limit is set to height, we can define the maximum
    height of the regression tree.
    leaf_size : int
    If limit is set to leaf_size, we can define the maximum
    number of samples in a leaf for the regression tree.
    limit : string
    Setting this value will determine if a restriction on
    the regression tree is set. Either tree height or leaf
    size. If neither is chosen tree will grow until
    each sample is a leaf.
    Attributes
    -----------
    tree : {dictionary, binary tree-like}
    A binary regression tree where each node is a dictionary.
    sse : float
    Sum of squared error. The lowest error rate is used to
    determine best split at a branch or root node.
    """
    def __init__(self, X, y, max_height=float('inf'), leaf_size=1, limit=None):
        self.X = X
        self.y = y
        self.max_height = max_height
        self.leaf_size = leaf_size
        self.limit = limit
        self.tree = None

    def fit(self):
        """Fit training data.
        Parameters
        -----------
        The required parameters are the datasets.
        Parameters are defined in the class constructor (__init__)
        by the request of the prompt.
        Returns
        -----------
        self : object
        """
        # Call build_tree function to build regression tree.
        self.tree = self.build_tree(self.X, self.y, 0)
        return self

    def build_tree(self, X, y, current_height):
        """Builds binary regression tree.
        Parameters
        -----------
        X : {array-like}, shape = [num_samples, num_features]
        Training vectors, where num_samples is the number of
        samples and num_features is the number of features.
        Needs to be passed as a param as nodes created recursively
        and nodes made further should have a smaller sample.
        y : {array-like}, shape = [num_samples]
        Target values. Needs to be passed as a param as nodes
        created recursively and nodes made further should have
        a smaller sample.
        current_height : float
        Used to put current height in node.
        Returns
        -----------
        tree : {dictionary}
        """
        num_samples, num_features = X.shape

        # If limit is set to tree height check to see if it's been
        # reached. If reached, make current node a leaf node.
        if self.limit == "height" and (current_height >= self.max_height):
            return {
                'node_type': 'leaf',
                'height': current_height,
                'sample(s)': X,
                'value(s)': y,
                'predict_value' : np.mean(y)
            }

        # If limit is set to leaf size check to see if it's been
        # reached. If reached, make current node a leaf node.
        elif self.limit == "leaf_size" and (num_samples <= self.leaf_size):
            return {
                'node_type': 'leaf',
                'height': current_height,
                'sample(s)': X,
                'value(s)': y,
                'predict_value': np.mean(y)
            }

        else:
            # If only one sample remains, cannot split, make leaf node.
            if num_samples == 1:
                return {
                    'node_type': 'leaf',
                    'height': current_height,
                    'sample(s)': X,
                    'value(s)': y,
                    'predict_value': np.mean(y)
                }
            else:
                # If leaf node requirements not met, then it's a branch (or root).
                # Create a left tree and right tree split and recurse
                # until all leaf nodes are created.
                split = self.split(X, y, num_features)
                left_tree = self.build_tree(*split['left_split'], current_height + 1)
                right_tree = self.build_tree(*split['right_split'], current_height + 1)
                return {
                    'node_type': 'root' if current_height == 0 else 'branch',
                    'height': current_height,
                    'sample_split' : split['sample_split'],
                    'feature': split['feature'],
                    'feature_value': split['feature_value'],
                    'left': left_tree,
                    'right': right_tree
                }

    def split(self, X, y, num_features):
        """Finds best split for branch or root node using sum of squared error.
        Parameters
        -----------
        X : {array-like}
        Set or subset of training vectors.
        y : {array-like}, shape = [num_samples]
        Set or subset of target values.
        num_features : int
        Number of feature for current set or subset of data.
        Returns
        -----------
        best_split : {node information, dictionary}
        """
        best_split = None
        best_sse = float('inf')

        # Double for loop to compare every feature of every sample given
        # by X parameter.
        for i in range(num_features):
            j = 0
            for k in X[:, i]:
                left_split = list()
                right_split = list()

                # Inner for loop to create leaf and right split based
                # on current feature value.
                for l, x in enumerate(X[:, i]):
                    if x <= k:
                        left_split.append(l)
                    else:
                        right_split.append(l)

                # If there is a split calculate sum of squared error (sse).
                # If current split sse is lower than stored best_sse, update
                # best_sse value and update best_split information
                # to sample with best sse split.
                if left_split and right_split:
                    ly = y[left_split]
                    ry = y[right_split]
                    #Sum of Squared Error
                    sse = np.sum((ly - np.mean(ly))**2) + np.sum((ry - np.mean(ry))**2)
                    if sse < best_sse:
                        best_sse = sse
                        best_split = {
                            'sample_split': (X[j]),
                            'feature': i,
                            'feature_value': k,
                            'left_split': (X[left_split], ly),
                            'right_split': (X[right_split], ry),
                        }
                j += 1
        # Return node information with sample & features that have best split.
        return best_split

    def print_tree(self, node=None, depth=0, space=1):
        """Print entire regression tree.
        Parameters
        -----------
        node : {tree, tree-nodes, dictionary}
        Used to grab node and use node to print node information.
        depth : int
        Used to infer depth and help space out print statements based
        on tree depth.
        space : int
        Parameter to increase spacing of each node printed in tree,
        if desired.
        Returns
        -----------
        None
        """
        if node is None:
            node = self.tree
        if node['node_type'] == 'leaf':
            print(f"{' '*depth*space}Node: type = {node['node_type']}, height = {node['height']}, samples = {np.array2string(node['sample(s)'], separator=' ', formatter={'all': lambda x: str(x)}).replace("\n", "")}, values = {node['value(s)']},  predict_value = {node['predict_value']}")
        else:
            print(f"{' '*depth*space}Node: type = {node['node_type']}, height = {node['height']}, sample split = {node['sample_split']},  feature {node['feature']}, <= {node['feature_value']}")
            print(f"{' '*depth*space}Left:")
            self.print_tree(node['left'], depth + 1, space)
            print(f"{' '*depth*space}Right:")
            self.print_tree(node['right'], depth + 1, space)

    def tree_traversal(self, node, x, use, path):
        """Traverses regression tree.
        If used in predict it returns predicted value for a sample.
        If used in decision_path it returns list of strings with
        information about path taken for specified sample.
        Parameters
        -----------
        node : {current node, dictionary}
        x : {sample vector data}
        use : {string}
        Define use to return predictive value or
        path information.
        path : {list}
        A list to recursively update path list
        Returns
        -----------
        predict_value : {float}
        or
        path : {list of strings}
        """
        if node['node_type'] == 'leaf':
            if use == 'predict':
                return node['predict_value']
            elif use == 'decision_path':
                temp = "node: " + node['node_type'] + ", predict value: " + str(node['predict_value'])
                path.append(temp)
                return path
        if x[node['feature']] <= node['feature_value']:
            if use == 'decision_path':
                temp = "node: " + node['node_type'] + " feature " + str(node['feature']) + " compared sample " + str(x[node['feature']]) +  " <= " + str(node['feature_value']) + " move left -> "
                path.append(temp)
            return self.tree_traversal(node['left'], x, use, path)
        else:
            if use == 'decision_path':
                temp = "node: " + node['node_type'] + " feature " + str(node['feature']) + " compared sample " + str(x[node['feature']]) +  " <= " + str(node['feature_value']) + " move left -> "
                path.append(temp)
            return self.tree_traversal(node['right'], x, use, path)

    def predict(self, X):
        """Return class label after tree traversal"""
        predicted = list()
        for x in X:
            predicted.append(self.tree_traversal(self.tree, x, 'predict', list()))
        return np.array(predicted)

    def decision_path(self, X, space=1):
        """Return path taken for class label"""
        paths = list()
        for x in X:
            paths.append(self.tree_traversal(self.tree, x, 'decision_path', list()))
        self.print_paths(X, paths, 0, space)
        return self

    def print_paths(self, X, paths, depth, space):
        """Helper function to print decision path, similar to print_tree"""
        for i in range(len(paths)):
            print("For sample: ",X[i])
            for j in paths[i]:
                print(f"{' '*depth*space} {j}")
                depth += 1
            depth = 0
        return self
