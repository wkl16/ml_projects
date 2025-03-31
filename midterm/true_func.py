import time
import matplotlib.pyplot as plt
import numpy as np
from regressionTree import RegressionTree as r_tree
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error

def Task_4_Next_State(x1, x2):
    cool_x1 = (0.9 * x1) - (0.2 * x2)
    cool_x2 = (0.2 * x1) + (0.9 * x2)
    return cool_x1, cool_x2

def Task_4_Range(startx1, startx2, endx1, endx2, num_points):
    x1s = np.random.uniform(low=startx1, high=endx1, size=(num_points, 1))
    x2s = np.random.uniform(low=startx2, high=endx2, size=(num_points, 1))

    cool_x1s = np.empty((0,1))
    cool_x2s = np.empty((0,1))

    for i in range(num_points):
        cool_x1, cool_x2 = Task_4_Next_State(x1s[i], x2s[i])

        cool_x1s = np.vstack((cool_x1s, [cool_x1]))
        cool_x2s = np.vstack((cool_x2s, [cool_x2]))

    return x1s, x2s, cool_x1s, cool_x2s


def Task_4_xz(x):

    z = 0
    for _ in range(20):
        if x > 1:
            x = 0
        else:
            x = x + 0.2
        z = z + x

    return x, z     


def Task_4_xz_generate(startx1, startx2, endx1, endx2, num_points):
    x1s = np.random.uniform(low=startx1, high=endx1, size=(num_points, 1))
    # x2s = np.random.uniform(low=startx2, high=endx2, size=(num_points, 1))

    cool_x1s = np.empty((0,1))
    cool_x2s = np.empty((0,1))

    for i in range(num_points):
        cool_x1, cool_x2 = Task_4_xz_v2(x1s[i])

        cool_x1s = np.vstack((cool_x1s, [cool_x1]))
        cool_x2s = np.vstack((cool_x2s, [cool_x2]))

    return x1s, x2s, cool_x1s, cool_x2s
    

if __name__ == "__main__":
    start_range =  -5
    end_range   =   5
    init_x2     = 0.5
    init_x1     = 1.5

    x1s, x2s, cool_x1s, cool_x2s = Task_4_Range(-5, -5, 5, 5, 1000)
    cool_x2s = cool_x2s.ravel()
    cool_x1s = cool_x1s.ravel()

    init_x = 2
    init_z = 0

    xs, zs, cool_xs, cool_zs = Task_4_xz_generate(-3, 0, 3, 15, 1000)
    cool_xs = cool_xs.ravel()
    cool_zs = cool_zs.ravel()

    # xs, zs, cool_xs, cool_zs = Task_4_xz_v2_generate(-3, 0, 3, 15, 1000)
    # cool_xs = cool_xs.ravel()
    # cool_zs = cool_zs.ravel()

    """"""
    x1s_train, x1s_test, cool_x1s_train, cool_x1s_test = train_test_split(
                                                         x1s, cool_x1s, test_size = 0.2, random_state=42
                                                         )
    x2s_train, x2s_test, cool_x2s_train, cool_x2s_test = train_test_split(
                                                         x2s, cool_x2s, test_size = 0.2, random_state=42
                                                         )
    
    xs_train, xs_test, cool_xs_train, cool_xs_test = train_test_split(
                                                         xs, cool_xs, test_size = 0.2, random_state = 42
                                                         ) 
    zs_train, zs_test, cool_zs_train, cool_zs_test = train_test_split(
                                                         xs, cool_xs, test_size = 0.2, random_state = 42
                                                         ) 
    
    """"""
    ### No limit
    x1_reg_tree = r_tree(X=x1s_train, y=cool_x1s_train, max_height=float('inf'), leaf_size=1, limit=None)
    x2_reg_tree = r_tree(X=x2s_train, y=cool_x2s_train, max_height=float('inf'), leaf_size=1, limit=None)
    x1_reg_tree.fit()
    x2_reg_tree.fit()
    
    """"""
    ### Height Limit
    x1_reg_t_ht = r_tree(X=x1s_train, y=cool_x1s_train, max_height=3, leaf_size=1, limit="Height")
    x2_reg_t_ht = r_tree(X=x2s_train, y=cool_x2s_train, max_height=3, leaf_size=1, limit="Height")
    x1_reg_t_ht.fit()
    x2_reg_t_ht.fit()

    """"""
    ### Leaf Limit
    x1_reg_t_lf = r_tree(X=x1s_train, y=cool_x1s_train, max_height=float('inf'), leaf_size=5, limit="leaf_size")
    x2_reg_t_lf = r_tree(X=x2s_train, y=cool_x2s_train, max_height=float('inf'), leaf_size=5, limit="leaf_size")
    x1_reg_t_lf.fit()
    x2_reg_t_lf.fit()

    """"""
    ### 3 point program
    x_reg_tree = r_tree(X=xs_train, y=cool_xs_train, max_height=float('inf'), leaf_size=1, limit=None)
    z_reg_tree = r_tree(X=zs_train, y=cool_zs_train, max_height=float('inf'), leaf_size=1, limit=None)
    x_reg_tree.fit()
    z_reg_tree.fit()

    """"""

    # x1_pred = x1_reg_tree.predict(x1s_test)
    # x2_pred = x2_reg_tree.predict(x2s_test)

    # print("x1 tests", x1s_test)
    # print("x2 tests", x2s_test)
    # print()
    # print()

    # print("x1 Prediction: ", x1_pred)
    # print()
    # print("x2 Prediction: ", x2_pred)
    # print()
    # print()

    """"""""""""""
    print("x1 = 1.5, x2 = 0.5 Prediction analysis, No Limit")

    x1_init_pred = np.array([[init_x1]])
    x2_init_pred = np.array([[init_x2]])

    temp_x1 = init_x1
    temp_x2 = init_x2

    for i in range(20):
        x1_next = x1_reg_tree.predict([[temp_x1]])
        x2_next = x2_reg_tree.predict([[temp_x1]])
        x1_init_pred = np.vstack((x1_init_pred, [x1_next]))
        x2_init_pred = np.vstack((x2_init_pred, [x2_next]))
        temp_x1 = x1_next
        temp_x2 = x2_next
    
    x1_init_pred = x1_init_pred.ravel()
    x2_init_pred = x2_init_pred.ravel()

    """"""""""""""
    x1_actual = np.array([init_x1])
    x2_actual = np.array([init_x2])

    temp_x1 = init_x1
    temp_x2 = init_x2

    for i in range(20):
        x1_next, x2_next = Task_4_Next_State(temp_x1, temp_x2)
        x1_actual = np.append(x1_actual, x1_next)
        x2_actual = np.append(x2_actual, x2_next)
        temp_x1 = x1_next
        temp_x2 = x2_next

    
    print("x1 = 1.5 pred: ", x1_init_pred)
    print("x1 Actual    : ", x1_actual)
    print("x2 = 0.5 pred: ", x2_init_pred)
    print("x2 Actual    : ", x2_actual)
    print()
    print()

    """"""""""""""
    print("x1 = 1.5, x2 = 0.5 Prediction analysis, Height Limit")

    x1_init_pred = np.array([[init_x1]])
    x2_init_pred = np.array([[init_x2]])

    temp_x1 = init_x1
    temp_x2 = init_x2

    for i in range(20):
        x1_next = x1_reg_t_ht.predict([[temp_x1]])
        x2_next = x1_reg_t_ht.predict([[temp_x1]])
        x1_init_pred = np.vstack((x1_init_pred, [x1_next]))
        x2_init_pred = np.vstack((x2_init_pred, [x2_next]))
        temp_x1 = x1_next
        temp_x2 = x2_next
    
    x1_init_pred = x1_init_pred.ravel()
    x2_init_pred = x2_init_pred.ravel()
    
    """"""""""""""
    x1_actual = np.array([init_x1])
    x2_actual = np.array([init_x2])

    temp_x1 = init_x1
    temp_x2 = init_x2

    for i in range(20):
        x1_next, x2_next = Task_4_Next_State(temp_x1, temp_x2)
        x1_actual = np.append(x1_actual, x1_next)
        x2_actual = np.append(x2_actual, x2_next)
        temp_x1 = x1_next
        temp_x2 = x2_next

    
    print("x1 = 1.5 pred: ", x1_init_pred)
    print("x1 Actual    : ", x1_actual)
    print("x2 = 0.5 pred: ", x2_init_pred)
    print("x2 Actual    : ", x2_actual)
    print()
    print()

    """"""""""""""
    print("x1 = 1.5, x2 = 0.5 Prediction analysis, Leaf Limit")

    x1_init_pred = np.array([[init_x1]])
    x2_init_pred = np.array([[init_x2]])

    temp_x1 = init_x1
    temp_x2 = init_x2

    for i in range(20):
        x1_next = x1_reg_t_lf.predict([[temp_x1]])
        x2_next = x1_reg_t_lf.predict([[temp_x1]])
        x1_init_pred = np.vstack((x1_init_pred, [x1_next]))
        x2_init_pred = np.vstack((x2_init_pred, [x2_next]))
        temp_x1 = x1_next
        temp_x2 = x2_next
    
    x1_init_pred = x1_init_pred.ravel()
    x2_init_pred = x2_init_pred.ravel()
    
    """"""""""""""
    x1_actual = np.array([init_x1])
    x2_actual = np.array([init_x2])

    temp_x1 = init_x1
    temp_x2 = init_x2

    for i in range(20):
        x1_next, x2_next = Task_4_Next_State(temp_x1, temp_x2)
        x1_actual = np.append(x1_actual, x1_next)
        x2_actual = np.append(x2_actual, x2_next)
        temp_x1 = x1_next
        temp_x2 = x2_next

    
    print("x1 = 1.5 pred: ", x1_init_pred)
    print("x1 Actual    : ", x1_actual)
    print("x2 = 0.5 pred: ", x2_init_pred)
    print("x2 Actual    : ", x2_actual)

    print("x = 2, z = 0 Prediction analysis, No Limit")

    x_init_pred = np.array([[init_x]])
    z_init_pred = np.array([[init_z]])

    temp_x = init_x
    temp_z = init_z

    for i in range(20):
        x_next = x_reg_tree.predict([[temp_x]])
        z_next = z_reg_tree.predict([[temp_z]])
        x_init_pred = np.vstack((x_init_pred, [x_next]))
        z_init_pred = np.vstack((z_init_pred, [z_next]))
        temp_x = x_next
        temp_z = z_next
    
    x_init_pred = x_init_pred.ravel()
    z_init_pred = z_init_pred.ravel()

    """"""""""""""
    x_actual = np.array([init_x])
    z_actual = np.array([init_z])

    temp_x = init_x
    temp_z = init_z

    for i in range(20):
        x_next, z_next = Task_4_xz_v2(temp_x)
        print(x_next)
        x_actual = np.append(x_actual, x_next)
        z_actual = np.append(z_actual, z_next)
        temp_x = x_next
        temp_z = z_next

    
    print("x = 2 pred: ", x_init_pred)
    print("x Actual    : ", x_actual)
    print("z = 0 pred: ", z_init_pred)
    print("z Actual    : ", z_actual)
    print()
    print()


    
    
