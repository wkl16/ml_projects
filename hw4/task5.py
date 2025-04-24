## Training using Dropout Regularization

import time
import torch
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from generate_data import generate_df_from_reviews
from generate_data import preprocessor
from generate_data import tokenizer_porter

# functions
def process_data():
    start_time = time.time()

    # Generate DataFrame Object where data is stored
    print("\n1. Loading dataset...")
    df = generate_df_from_reviews()
    print(f"Total number of reviews in dataset: {len(df)}")
    
    # 70-30 split
    # X_train = df.loc[:35000, 'review'].values
    # X_test = df.loc[15000:, 'review'].values
    # y_train = df.loc[:35000, 'sentiment'].values
    # y_test = df.loc[15000:, 'sentiment'].values
    
    # 
    split_idx = 35000
    X_train = df.loc[:split_idx-1, 'review'].values
    X_test = df.loc[split_idx:, 'review'].values
    y_train = df.loc[:split_idx-1, 'sentiment'].values
    y_test = df.loc[split_idx:, 'sentiment'].values
    # data leakage due to overlap from row 15000 through 35000 between training and testing data! 

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    
    # Limit max features to roughly half of what it 
    # would originally generate with this vectorizer
    # Change max_features to include more features for
    # processing. 50000 features went to about 2-3 hours
    # Original size from a very initial test was around
    # 100000 features. If you change this, change the scale
    # of the FNN and change the input in net.forward()
    # to also match this number.
    tfidf = TfidfVectorizer(max_features=10000, strip_accents=None, lowercase=False, preprocessor=preprocessor, tokenizer=tokenizer_porter, stop_words='english')
    
    # Vectorize the review data and turn it into something
    # that the TensorDataset can ingest
    # Also vectorize test data to test on finished model
    tfidf_reviews = tfidf.fit_transform(X_train).toarray()
    
    # tfidf_testing = tfidf.fit_transform(X_test).toarray()
    tfidf_testing = tfidf.transform(X_test).toarray() 
    # doesn't need to fit the vectorizer again, test data should be vectorized using the same vocab and transform learned from the training data

    tfidf_reviews_norm = (tfidf_reviews - np.mean(tfidf_reviews)) / np.std(tfidf_reviews)
    tfidf_testing_norm = (tfidf_testing - np.mean(tfidf_testing)) / np.std(tfidf_testing)
    # print(tfidf_reviews_norm[1])
    # print(tfidf_reviews_norm.shape)

    # Float for the reviews, as the array is likely to have floats rather than ints
    # long for the sentiments to properly gauge positive or negative
    tfidf_reviews_norm = torch.from_numpy(tfidf_reviews_norm).float()
    tfidf_testing_norm = torch.from_numpy(tfidf_testing_norm).float()
    y_train = torch.from_numpy(y_train).long()
    

    # Initialize the dataset and load it into the dataloader
    train_ds = TensorDataset(tfidf_reviews_norm, y_train)
    review_data = DataLoader(train_ds, batch_size=100, shuffle=True)


    # initialize FNN. Model based off of Linear ReLU from class
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
    
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    L = torch.nn.CrossEntropyLoss()

    # Train. It's doing something and it takes a while. I got to take a shower and it barely finished the
    # Second epoch. So to finish 5 epochs, likely 1 hour and 30 minutes.
    print("\n5. starting training...")
    print("Epoch | Training Loss | Test Accuracy | Time")
    print("-" * 40)

    acc_list = []
    start_time_training = time.time() 
    
    for epoch in range(8):
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
    total_training_time = time.time() - start_time_training

    print(f"\n Training completed in {total_time/60:.2f} minutes")
    print(f"Final test accuracy: {acc:.4f}")
    '''
    torch.onnx.export(
        net,
        (tfidf_reviews_norm,),
        "Movie_Review_Model.onnx",
        input_names=["reviews"],
        dynamo=True
    )
    '''
    return acc_list, total_training_time

 
def pre_process_data():

    # Generate DataFrame Object where data is stored
    print("\n1. Loading dataset...")
    df = generate_df_from_reviews()
    print(f"Total number of reviews in dataset: {len(df)}")

    # 70-30 split
    # X_train = df.loc[:35000, 'review'].values
    # X_test = df.loc[15000:, 'review'].values
    # y_train = df.loc[:35000, 'sentiment'].values
    # y_test = df.loc[15000:, 'sentiment'].values

    #
    split_idx = 35000
    X_train = df.loc[:split_idx - 1, 'review'].values
    X_test = df.loc[split_idx:, 'review'].values
    y_train = df.loc[:split_idx - 1, 'sentiment'].values
    y_test = df.loc[split_idx:, 'sentiment'].values
    # data leakage due to overlap from row 15000 through 35000 between training and testing data!

    #print(f"Training set size: {len(X_train)}")
    #print(f"Test set size: {len(X_test)}")

    features = [50000]
    learning_rate = [0.01]
    L2 = [0.01]
    y_train = torch.from_numpy(y_train).long()

    for i in features:
        tfidf = TfidfVectorizer(max_features=i, strip_accents=None, lowercase=False,
                                preprocessor=preprocessor,
                                tokenizer=tokenizer_porter, stop_words='english')

        # Vectorize the review data and turn it into something
        # that the TensorDataset can ingest
        # Also vectorize test data to test on finished model
        tfidf_reviews = tfidf.fit_transform(X_train).toarray()

        # tfidf_testing = tfidf.fit_transform(X_test).toarray()
        tfidf_testing = tfidf.transform(X_test).toarray()
        # doesn't need to fit the vectorizer again, test data should be vectorized using the same vocab and transform learned from the training data

        tfidf_reviews_norm = (tfidf_reviews - np.mean(tfidf_reviews)) / np.std(tfidf_reviews)
        tfidf_testing_norm = (tfidf_testing - np.mean(tfidf_testing)) / np.std(tfidf_testing)
        # print(tfidf_reviews_norm[1])
        # print(tfidf_reviews_norm.shape)

        # Float for the reviews, as the array is likely to have floats rather than ints
        # long for the sentiments to properly gauge positive or negative
        tfidf_reviews_norm = torch.from_numpy(tfidf_reviews_norm).float()
        tfidf_testing_norm = torch.from_numpy(tfidf_testing_norm).float()

        # Initialize the dataset and load it into the dataloader
        train_ds = TensorDataset(tfidf_reviews_norm, y_train)
        review_data = DataLoader(train_ds, batch_size=100, shuffle=True)
        #fnn(review_data, tfidf_testing_norm, y_test, i, 0.1, 0)
        for j in learning_rate:
            for k in L2:
                fnn(review_data, tfidf_testing_norm, y_test, i, j, k)
    '''
    torch.onnx.export(
        net,
        (tfidf_reviews_norm,),
        "Movie_Review_Model.onnx",
        input_names=["reviews"],
        dynamo=True
    )
    '''


# using max features = 10000, epochs = 8 for all

# import FNN from task 2
base_acc, base_time = process_data()

# using PyTorch function dropout
# may tune the dropout probability for each layer

# Create a single dropout model and compare its performance to the baseline model
def gen_single_dropout():
    # layers
    net_single_dropout = torch.nn.Sequential(
        torch.nn.Linear(10000, 5000),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(5000, 5000),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(5000, 2500),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),
        torch.nn.Linear(2500, 2),
        torch.nn.Softmax(dim=1))
    
    optimizer_single_dropout = torch.optim.SGD(net_single_dropout.parameters(), lr=0.1)
    L_single_dropout = torch.nn.CrossEntropyLoss()

# Create a set of at least 5 different dropout models, train them using bagging 


## plotting
# 3 plots comparing 1. baseline 2. single dropout 3. bagging w/ 5 dropout models

# time cost 
# table of 3 model time to train to 8 epochs
print(f'\n Training time, base FNN: {base_time/60:.2f} minutes')

# accuracy comparison
# x axis = epochs (8)
epochs = list(range(1,9))

# y axis = accuracy
# set y axis to be consistent across all plots?
plt.plot(epochs, base_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()