import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

def load_dataset(dataset_dir, d, n):
    """Load dataset from text file, split into train/test."""
    filepath = os.path.join(dataset_dir, f"samples_{d}d_{n}s.txt")
    data = np.loadtxt(filepath, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    split_idx = int(0.7 * len(y))
    return (X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:])

def evaluate_svc_scalability(dataset_dir, dimensions, sample_sizes, eta=0.01, n_iter=50, C=1.0):
    """Evaluate LinearSVC scalability across dimensions and sample sizes."""
    results = []
    for d in dimensions:
        for n in sample_sizes:
            print(f"Processing d={d}, n={n}")
            X_train, X_test, y_train, y_test = load_dataset(dataset_dir, d, n)

            model = LinearSVC(C=C, random_state=1,  max_iter=n, dual=False)

            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start
            
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            results.append({
                'Dimensions (d)': d,
                'Sample Size (n)': n,
                'Training Time (s)': train_time,
                'Accuracy': acc
            })
    return pd.DataFrame(results)

def plot_scalability_results(results_df, output_dir='.'):
    """Generate training time and accuracy plots."""
    # training time vs dimensions
    plt.figure(figsize=(8, 5))
    ax = results_df.pivot_table(values='Training Time (s)', index='Dimensions (d)', columns='Sample Size (n)').plot(marker='o')
    ax.set_yscale('log')
    plt.xlabel('Dimensions (d)')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Dimensions')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_time_vs_dims.png'))
    
    # training time vs sample size
    plt.figure(figsize=(8, 5))
    results_df.pivot_table(values='Training Time (s)', index='Sample Size (n)', columns='Dimensions (d)').plot(marker='o')
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time vs Sample Size')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_time_vs_samples.png'))
    
    # accuracy plot
    plt.figure(figsize=(8, 5))
    results_df.pivot_table(values='Accuracy', index='Dimensions (d)', columns='Sample Size (n)').plot(marker='o')
    plt.xlabel('Dimensions (d)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Dimensions')
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_dims.png'))

def analyze_and_print_conclusions(results_df):
    """Print key conclusions from results."""    
    # accuracy analysis
    acc_by_dim = results_df.groupby('Dimensions (d)')['Accuracy'].mean()
    print("Average accuracy across dimensions:\n" + acc_by_dim.to_string())

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    
    # data_dir_name = "dataset"
    # if not (os.path.exists(data_dir_name) and os.path.isdir(data_dir_name)):
    #     os.mkdir("dataset")
    
    # dataset_dir = os.path.join(current_dir, "dataset")
    dataset_dir = current_dir

    outs_dir_name = "Param_Set_2"
    if not (os.path.exists(outs_dir_name) and os.path.isdir(outs_dir_name)):
        os.mkdir("Param_Set_2")
    
    results_dir = os.path.join(current_dir, "Param_Set_3")


    # extract dimensions and sample sizes from filenames
    files = [f for f in os.listdir(dataset_dir) if f.startswith("samples_") and f.endswith("s.txt")]
    dimensions = sorted({int(f.split("_")[1].split('d')[0]) for f in files})
    sample_sizes = sorted({int(f.split("_")[2].split('s')[0]) for f in files})
    
    print(f"Dimensions: {dimensions}\nSample sizes: {sample_sizes}")
    
    results_df = evaluate_svc_scalability(dataset_dir, dimensions, sample_sizes)
    print("\nResults:")
    print(results_df.to_string(index = False))
    
    # accuracy analysis
    acc_by_dim = results_df.groupby('Dimensions (d)')['Accuracy'].mean()
    print("Average accuracy across dimensions:\n" + acc_by_dim.to_string())
    
    results_df.to_csv(os.path.join(results_dir, "svm_scalability_results.csv"), index=False)
    plot_scalability_results(results_df, results_dir)