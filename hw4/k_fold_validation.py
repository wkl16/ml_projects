import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from generate_data import generate_df_from_reviews, preprocessor, tokenizer_porter

INPUT_SIZE = 10000
def reset_weights(m):
    """
    Reset trainable parameters to avoid weight leakage between folds.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def prepare_data():
    df = generate_df_from_reviews()
    reviews = df['review'].values
    labels = df['sentiment'].values.astype(np.int64)

    tfidf = TfidfVectorizer(
        max_features=INPUT_SIZE,
        preprocessor=preprocessor,
        tokenizer=tokenizer_porter,
        stop_words='english')
    X = tfidf.fit_transform(reviews).toarray().astype(np.float32)
    X = (X - X.mean()) / X.std()
    
    return torch.from_numpy(X), torch.from_numpy(labels)

def create_model():
    return nn.Sequential(
        nn.Linear(INPUT_SIZE, INPUT_SIZE//2),
        nn.ReLU(),
        nn.Linear(INPUT_SIZE//2, INPUT_SIZE//2),
        nn.ReLU(),
        nn.Linear(INPUT_SIZE//2, INPUT_SIZE//2),
        nn.ReLU(),
        nn.Linear(INPUT_SIZE//2, INPUT_SIZE//4),
        nn.ReLU(),
        nn.Linear(INPUT_SIZE//4, 2)
    )

def train_fold(train_loader, test_loader, num_epochs, learning_rate):
    # Train and evaluate model on one fold of the data
    model = create_model()
    model.apply(reset_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print("Epoch | Training Loss | Test Accuracy | Time")
    print("-" * 40)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # update model weights using batches of data
        model.train()
        total_loss = 0.0
        batch_count = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        avg_loss = total_loss / batch_count

        # compute accuracy on test set
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, preds = torch.max(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
        accuracy = correct / total
        
        epoch_time = time.time() - epoch_start
        print(f"{epoch:5d} | {avg_loss:.6f} | {accuracy:.6f} | {epoch_time:.2f}s")
    
    return accuracy

def train_with_kfold(k_folds=5, num_epochs=5, batch_size=100, learning_rate=0.1):
    # Perform k-fold cross validation to assess model performance
    print("Loading and preparing data...")
    X_tensor, y_tensor = prepare_data()
    dataset = TensorDataset(X_tensor, y_tensor)
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = {}
    
    total_start = time.time()
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_tensor)):
        print(f"\nFold {fold+1}/{k_folds}")
        print("-" * 40)
        
        # Create data loaders for current fold's train and test sets
        train_loader = DataLoader(dataset, batch_size=batch_size, 
                        sampler=SubsetRandomSampler(train_idx))
        test_loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=SubsetRandomSampler(test_idx))
        
        accuracy = train_fold(train_loader, test_loader, num_epochs, learning_rate)
        fold_results[fold] = accuracy
    
    total_time = time.time() - total_start
    print("\nK-FOLD CROSS VALIDATION RESULTS")
    print("-" * 40)
    print(f"Average Accuracy: {np.mean(list(fold_results.values())):.4f}")
    print(f"Total Time: {total_time/60:.2f} minutes")
    
    return np.mean(list(fold_results.values()))

def tune_kfold_parameter(num_epochs=5, batch_size=100, learning_rate=0.1):
    
    # Experiment with different k values to find optimal number of folds
    print("Starting K-Fold Parameter Tuning")
    print("-" * 40)
    
    results = {}
    for k in range(9, 13):
        print(f"\nTesting with k={k} folds")
        print("-" * 40)
        results[k] = train_with_kfold(k, num_epochs, batch_size, learning_rate)
    
    print("\nK-FOLD PARAMETER TUNING RESULTS")
    print("-" * 40)
    print("k | Average Accuracy")
    print("-" * 40)
    for k, acc in results.items():
        print(f"{k:2d} | {acc:.6f}")
    
    best_k = max(results, key=results.get)
    print(f"\nBest k value: {best_k} with accuracy: {results[best_k]:.6f}")

if __name__ == "__main__":
    tune_kfold_parameter(num_epochs=5, batch_size=100, learning_rate=0.1)
