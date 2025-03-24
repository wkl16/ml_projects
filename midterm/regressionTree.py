import numpy as np

class RegressionTree:
    def __init__(self, X, y, max_height=float('inf'), leaf_size=1, limit=None):
        self.X = X
        self.y = y
        self.max_height = max_height
        self.leaf_size = leaf_size
        self.limit= limit
        self.tree = None

    def fit(self):
        self.tree = self.build_tree(self.X, self.y, 0)
        return self.tree

    def build_tree(self, X, y, current_height):
        num_samples, num_features = X.shape

        if self.limit == "height" and (current_height >= self.max_height):
            return {
                'node_type': 'leaf',
                'height': current_height,
                'sample(s)': X,
                'value(s)': y,
                'predict_value' : np.mean(y)
            }

        elif self.limit == "leaf_size" and (num_samples <= self.leaf_size):
            return {
                'node_type': 'leaf',
                'height': current_height,
                'sample(s)': X,
                'value(s)': y,
                'predict_value': np.mean(y)
            }

        else:
            if num_samples == 1:
                return {
                    'node_type': 'leaf',
                    'height': current_height,
                    'sample(s)': X,
                    'value(s)': y,
                    'predict_value': np.mean(y)
                }
            else:
                best_split = self.split(X, y, num_features)
                left_tree = self.build_tree(*best_split['left_node'], current_height + 1)
                right_tree = self.build_tree(*best_split['right_node'], current_height + 1)
                return {
                    'node_type': 'root' if current_height == 0 else 'branch',
                    'height': current_height,
                    'feature': best_split['feature'],
                    'feature_value': best_split['feature_value'],
                    'left': left_tree,
                    'right': right_tree
                }

    def split(self, X, y, num_features):
        best_split = None
        best_sse = float('inf')

        for i in range(num_features):
            for j in X[:, i]:
                left_split = list()
                right_split = list()

                for k, x in enumerate(X[:, i]):
                    if x <= j:
                        left_split.append(k)
                    else:
                        right_split.append(k)

                if left_split and right_split:
                    left_y = y[left_split]
                    right_y = y[right_split]
                    sse = self.sse(left_y, right_y)
                    if sse < best_sse:
                        best_sse = sse
                        best_split = {
                            'feature': i,
                            'feature_value': j,
                            'left_node': (X[left_split], left_y),
                            'right_node': (X[right_split], right_y),
                        }
        return best_split

    #Sum of Squares Error
    def sse(self, lty, rty):
        lt_sse = np.sum((lty - np.mean(lty))**2)
        rt_sse = np.sum((rty - np.mean(rty))**2)
        return lt_sse + rt_sse

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.tree
        if node['node_type'] == 'leaf':
            print(f"{' ' * depth*2}Leaf Node: height = {node['height']}, samples = {node['sample(s)']}, values = {node['value(s)']},  predict_value = {node['predict_value']}")
        else:
            print(f"{' ' * depth*2}Node: type = {node['node_type']}, height = {node['height']}, feature {node['feature']} <= {node['feature_value']}")
            print(f"{' ' * depth*2}Left:")
            self.print_tree(node['left'], depth + 1)
            print(f"{' ' * depth*2}Right:")
            self.print_tree(node['right'], depth + 1)

    def tree_traversal(self, node, x, use, path):
        if node['node_type'] == 'leaf':
            if use == 'predict':
                return node['predict_value']
            elif use == 'decision_path':
                temp = "node: " + node['node_type'] + ", predict value: " + str(node['predict_value'])
                path.append(temp)
                return path
        if x[node['feature']] <= node['feature_value']:
            temp = "node: " + node['node_type'] + " feature " + str(node['feature']) + " compared sample " + str(x[node['feature']]) +  " <= " + str(node['feature_value']) + " move left -> "
            path.append(temp)
            return self.tree_traversal(node['left'], x, use, path)
        else:
            temp = "node: " + node['node_type'] + " feature " + str(node['feature']) + " compared sample " + str(x[node['feature']]) +  " > " + str(node['feature_value']) + " move right -> "
            path.append(temp)
            return self.tree_traversal(node['right'], x, use, path)

    def predict(self, X):
        predicted = list()
        for x in X:
            predicted.append(self.tree_traversal(self.tree, x, 'predict', list()))
        return np.array(predicted)

    def decision_path(self, X):
        paths = list()
        for x in X:
            paths.append(self.tree_traversal(self.tree, x, 'decision_path', list()))
        self.print_paths(X, paths, 0)
        return self

    def print_paths(self, X, paths, depth):
        for i in range(len(paths)):
            print("For sample: ",X[i])
            for j in paths[i]:
                print(f"{' ' * depth*2} {j}")
                depth += 1
            depth = 0
        return self