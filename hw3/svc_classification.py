import numpy as np
from generate_data import generate_data_numbers, generate_data_fashion
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import time

def perform_grid_search(X_train, X_test, y_train, y_test, kernel, param_grid, pca_components=100):
    """
    Perform grid search for a specific kernel and parameter grid
    """
    # Create a pipeline with PCA, scaling, and SVC
    pipeline = Pipeline([
        ('pca', PCA(n_components=pca_components)),
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel=kernel, random_state=1, max_iter=10000))
    ])
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=4,
        scoring='accuracy'
    )
    
    # Time the training
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Get best results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    test_score = grid_search.score(X_test, y_test)
    
    return {
        'best_params': best_params,
        'best_cv_score': best_score,
        'test_score': test_score,
        'training_time': training_time
    }

def train_and_evaluate_svc(dataset):
    # Load appropriate dataset
    if dataset == 'numbers':
        X_train, X_test, y_train, y_test = generate_data_numbers()
    else:
        X_train, X_test, y_train, y_test = generate_data_fashion()
    
    print(f"\nPerforming grid search for {dataset} dataset...")
    
    # Define parameter grids for each kernel
    param_range_C = [0.01, 0.1, 1.0, 10.0]
    param_range_gamma = [0.01, 0.1, 1.0, 10.0]
    param_range_degree = [2, 3, 4, 5]
    
    param_grids = {
        'linear': {
            'svc__C': param_range_C
        },
        'rbf': {
            'svc__C': param_range_C,
            'svc__gamma': param_range_gamma
        },
        'poly': {
            'svc__C': param_range_C,
            'svc__gamma': param_range_gamma,
            'svc__degree': param_range_degree
        }
    }
    
    results = {}
    
    # Perform grid search for each kernel
    for kernel in ['linear', 'rbf', 'poly']:
        print(f"\nTraining {kernel} kernel...")
        results[kernel] = perform_grid_search(
            X_train, X_test, y_train, y_test,
            kernel,
            param_grids[kernel]
        )
    
    # Print results
    print(f"\nResults for {dataset} dataset:")
    print("=" * 80)
    for kernel, result in results.items():
        print(f"\n{kernel.upper()} Kernel Results:")
        print(f"Best parameters: {result['best_params']}")
        print(f"Best cross-validation accuracy: {result['best_cv_score']:.4f}")
        print(f"Test set accuracy: {result['test_score']:.4f}")
        print(f"Training time: {result['training_time']:.2f} seconds")
        print("-" * 40)

if __name__ == "__main__":
    train_and_evaluate_svc('numbers')
    train_and_evaluate_svc('fashion') 