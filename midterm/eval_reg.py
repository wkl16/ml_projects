import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from regressionTree import RegressionTree

# generate data for y = 0.8 sin(x - 1) over x ∈ [-3, 3]
np.random.seed(0)
n_samples = 100

X_all = np.random.uniform(low=-3, high=3, size=(n_samples, 1))
y_all = 0.8 * np.sin(X_all - 1)
y_all = y_all.ravel()  # Flatten to 1D

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

def get_max_depth(tree_dict):
    if tree_dict['node_type'] == 'leaf':
        return tree_dict['height']
    return max(get_max_depth(tree_dict['left']),
               get_max_depth(tree_dict['right']))

# train and evaluate a regressionTree
def train_and_eval(X_tr, y_tr, X_te, y_te,
                            limit_type=None, limit_value=None):
    """
    limit_type: "height", "leaf_size", or None
    limit_value: integer or None
    returns: (final_tree_height, test_mse, build_time, tree_object)
    """
    # decide how to configure the regressionTree
    if limit_type == "height":
        reg_tree = RegressionTree(X_tr, y_tr,
                                  max_height=limit_value,
                                  leaf_size=1,
                                  limit="height")
    elif limit_type == "leaf_size":
        reg_tree = RegressionTree(X_tr, y_tr,
                                  max_height=float('inf'),
                                  leaf_size=limit_value,
                                  limit="leaf_size")
    else:  # no limit
        reg_tree = RegressionTree(X_tr, y_tr,
                                  max_height=float('inf'),
                                  leaf_size=1,
                                  limit=None)

    start_time = time.time()
    reg_tree.fit()
    build_time = time.time() - start_time

    final_height = get_max_depth(reg_tree.tree)

    y_pred = reg_tree.predict(X_te)
    test_mse = mean_squared_error(y_te, y_pred)

    return final_height, test_mse, build_time, reg_tree

# 1) no limit 
print("\n1) No-limit scenario")
height_no_limit, mse_no_limit, time_no_limit, tree_no_limit = train_and_eval(
    X_train, y_train, X_test, y_test, limit_type=None, limit_value=None
)
print(f"No limit => tree height: {height_no_limit}, test error: {mse_no_limit:.6f}, build time:  {time_no_limit:.6f}s")
results_dict = {
    "No limit": (height_no_limit, mse_no_limit, time_no_limit),
}
# 2) height limit: 1/2 and 3/4 of 1)
half_height = max(1, int(height_no_limit / 2))            
three_quarter_height = max(1, int(0.75 * height_no_limit))
height_limits = [half_height, three_quarter_height]

print("2) Height-limited scenarios")
height_half, mse_half, time_half, _ = train_and_eval(X_train, y_train, X_test, y_test, limit_type="height", limit_value=half_height
)
for h_lim in height_limits:
    h, mse_, t, _ = train_and_eval(
        X_train, y_train, X_test, y_test,
        limit_type="height",
        limit_value=h_lim
    )
    print(f"Height limit={h_lim} => tree height={h}, MSE={mse_:.6f}, build time={t:.6f}s")
    results_dict[f"Height={h_lim}"] = (h_lim, mse_, t)
    
# 3) leaf size limits: 2, 4, 8
# testing: 2, 3,4,5,6,7, 8,9,10,12,16
print("3) Leaf-size-limited scenarios")
leaf_sizes = [2, 4, 8]

for ls in leaf_sizes:
    h_ls, mse_ls, t_ls, _ = train_and_eval( X_train, y_train, X_test, y_test, limit_type="leaf_size", limit_value=ls
    )
    print(f"Leaf size={ls} => tree height={h_ls}, MSE={mse_ls:.6f}, build time={t_ls:.6f}s")
    results_dict[f"LeafSize={ls}"] = (h_ls, mse_ls, t_ls)

# 4) display results in a single table
print("\nComparison Table")
print("{:<15} {:<15} {:<15} {:<15}".format("Scenario","Height","Test Error","Build Time"))

for scenario, (h, m, t) in results_dict.items():
    print("{:<15} {:<15} {:<15.6f} {:<15.6f}".format(scenario, str(h), m, t))
