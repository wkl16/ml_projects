## Training using Dropout Regularization

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

from generate_data import generate_df_from_reviews
from generate_data import preprocessor
from generate_data import tokenizer_porter

# functions
def preprocess_data(max_features = 10000):
    # code taken from earlier tasks but broken up for this task
    # input data, outputs DataLoader
    print("\n1. Loading dataset...")
    df = generate_df_from_reviews()
    print(f"Total number of reviews in dataset: {len(df)}")

    split_idx = 35000
    X_train = df.loc[:split_idx-1, 'review'].values
    X_test = df.loc[split_idx:, 'review'].values
    y_train = df.loc[:split_idx-1, 'sentiment'].values
    y_test = df.loc[split_idx:, 'sentiment'].values

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    tfidf = TfidfVectorizer(max_features=max_features, strip_accents=None, lowercase=False, preprocessor=preprocessor, tokenizer=tokenizer_porter, stop_words='english')
    
    tfidf_reviews = tfidf.fit_transform(X_train).toarray()
    tfidf_testing = tfidf.transform(X_test).toarray() 
    
    tfidf_reviews_norm = (tfidf_reviews - np.mean(tfidf_reviews)) / np.std(tfidf_reviews)
    tfidf_testing_norm = (tfidf_testing - np.mean(tfidf_testing)) / np.std(tfidf_testing)
    
    tfidf_reviews_norm = torch.from_numpy(tfidf_reviews_norm).float()
    tfidf_testing_norm = torch.from_numpy(tfidf_testing_norm).float()
    y_train = torch.from_numpy(y_train).long()
    y_test_tensor = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(tfidf_reviews_norm, y_train)
    review_data = DataLoader(train_ds, batch_size=100, shuffle=True)

    return review_data, tfidf_testing_norm, y_test_tensor

def gen_baseline():
    print("\n4. Initializing neural network...")
    net = torch.nn.Sequential(
        torch.nn.Linear(10000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 2500),
        torch.nn.ReLU(),
        torch.nn.Linear(2500, 2),
        torch.nn.Softmax(dim=1))
    
    return net

def gen_single_dropout():
    print("\n4. Initializing neural network...")
    net_single_dropout = torch.nn.Sequential(
        torch.nn.Linear(10000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Linear(5000, 2500),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(2500, 2),
        torch.nn.Softmax(dim=1))
    
    return net_single_dropout

def train(net, review_data, tfidf_testing_norm, y_test, epochs = 12):
    print("\n5. starting training...")
    print("Epoch | Training Loss | Test Accuracy | Time")
    print("-" * 40)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    L = torch.nn.CrossEntropyLoss()
    acc_list = []
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        net.train()
        total_loss = 0
        batch_count = 0

        for (x, y) in review_data:
            output = net.forward(x.view(-1,10000))
            loss = L(output, y)
            total_loss += loss.item()
            batch_count += 1
            loss.backward()
            optimizer.step()
            net.zero_grad()
        
        avg_loss = total_loss / batch_count

        net.eval()
        with torch.no_grad():
            y_pred = net(tfidf_testing_norm)
            y_pred = torch.argmax(y_pred, dim=1)
            acc = accuracy_score(y_test, y_pred)
            acc_list.append(acc)
            # print(acc)
            epoch_time = time.time() - epoch_start
            print(f"{epoch+1:5d} | {avg_loss:.6f} | {acc:.6f} | {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    return acc_list, total_time

# using max features = 10000, epochs = 12 for all
train_loader, X_test, y_test = preprocess_data()

# import FNN from task 2
print("\n=== Baseline Model Training ===")
baseline_net = gen_baseline()
base_acc, base_time = train(baseline_net, train_loader, X_test, y_test)
print(f"Baseline training time: {base_time/60:.2f} minutes, Final Acc: {base_acc[-1]:.4f}\n")

# using PyTorch function dropout
# may tune the dropout probability for each layer

# Create a single dropout model and compare its performance to the baseline model
print("=== Single Dropout Model Training ===")
single_net = gen_single_dropout()
single_acc, single_time = train(single_net, train_loader, X_test, y_test)
print(f"Single Dropout training time: {single_time/60:.2f} minutes, Final Acc: {single_acc[-1]:.4f}\n")

# Create a set of at least 5 different dropout models, train them using bagging 
print("=== Bagging 5 Dropout Models Training ===")
bag_accs = []
bag_times = []
for i in range(5):
    print(f"-- Training model {i+1} --")
    model = gen_single_dropout()
    acc, t = train(model, train_loader, X_test, y_test)
    bag_accs.append(acc)
    bag_times.append(t)


bag_acc = [sum(epoch_accs) / len(bag_accs) for epoch_accs in zip(*bag_accs)]
total_bag_time = sum(bag_times)
print(f"Bagging total time: {total_bag_time/60:.2f} minutes, Final Ensemble Avg Acc: {bag_acc[-1]:.4f}\n")

## plotting
# 3 plots comparing 1. baseline 2. single dropout 3. bagging w/ 5 dropout models

# time cost 
# table of 3 model time to train to 12 epochs
print("Training Time (minutes):")
print(f"Baseline: {base_time/60:.2f}")
print(f"Single Dropout: {single_time/60:.2f}")
print(f"Bagging (5 models): {total_bag_time/60:.2f}")

# accuracy comparison
# x axis = epochs (12)
epochs = list(range(1, len(base_acc) + 1))

# y axis = accuracy
# set y axis to be consistent across all plots?
plt.plot(epochs, base_acc, label="Baseline")
plt.plot(epochs, single_acc, label="Single Dropout")
plt.plot(epochs, bag_acc, label="Bagging (5 models)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Comparison")
plt.legend()
plt.show()