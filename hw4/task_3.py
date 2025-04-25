import pyprind
import pandas as pd
import numpy as np
import os
import sys
import re
import time
import logisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Don't think these will be used for me (Matt)
# But maybe for others
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Global seeds
torch.manual_seed(42)
np.random.seed(42)

# Global variables
csv_file_name = './movie_data.csv'
stop = stopwords.words('english')
porter = PorterStemmer()


# Generates a dataframe object from the movie reviews.
# returns
#   - df - A dataframe object containing all reviews and their sentiments
def generate_df_from_reviews():
    # If the csv file has not been generated yet,
    # generate from the aclImdb directory.
    if not (os.path.exists(csv_file_name)):
        current_dir = os.path.dirname(__file__)

        data_dir_name = "aclImdb"
        dataset_dir = os.path.join(current_dir, data_dir_name)
        labels = {'pos': 1, 'neg': 0}
        # pbar can be commented out
        pbar = pyprind.ProgBar(50000, stream=sys.stdout)
        df = pd.DataFrame()
        for s in ('test', 'train'):
            for l in ('pos', 'neg'):
                path = os.path.join(dataset_dir, s, l)
                for file in sorted(os.listdir(path)):
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                        txt = infile.read()
                    df_extension = pd.DataFrame([[txt, labels[l]]])
                    df = pd.concat([df, df_extension], ignore_index=True)
                    # pbar can be commented out
                    pbar.update()

        # Label columns
        df.columns = ['review', 'sentiment']
        # Shuffle
        df = df.reindex(np.random.permutation(df.index))
        # Store to csv
        df.to_csv(csv_file_name, index=False, encoding='utf-8')
    # Else, data has already been written, so read in from existing csv
    else:
        df = pd.read_csv(csv_file_name, encoding='utf-8')
        df = df.rename(columns={"0": "review", "1": "sentiment"})

    # Return resulting dataframe. Generated from seed 42
    return df


# Preprocessor to clean the data (From Textbook/slides)
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    # emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    # adding r for raw string to circumvent SyntaxWarning
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    return text


# Tokenizer (From Textbook/slides)
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


# Process the dataframe generated from the reviews
def process_data():

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

    features = [10000]
    learning_rate = [0.01]
    L2 = [0.01]
    lr_n_iter = [500]
    y_train = torch.from_numpy(y_train).long()
    y_test2 = torch.from_numpy(y_test).float()

    #update orignal process_data to loop through feature vectorization, and loop train logistic regression and FNN using
    # learning rate for both and L2 for FNN and number of iterations (lr
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

        # Float for the reviews, as the array is likely to have floats rather than ints
        # long for the sentiments to properly gauge positive or negative
        tfidf_reviews_norm = torch.from_numpy(tfidf_reviews_norm).float()
        tfidf_testing_norm = torch.from_numpy(tfidf_testing_norm).float()

        # Initialize the dataset and load it into the dataloader
        train_ds = TensorDataset(tfidf_reviews_norm, y_train)
        review_data = DataLoader(train_ds, batch_size=100, shuffle=True)
        for j in learning_rate:
            for m in lr_n_iter:
                run_lr(i, j, m, tfidf_reviews_norm, y_train, tfidf_testing_norm, y_test2)
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
# helper function to run logistic regression and grab runtime and accuracy.
def run_lr(features, mu, lr_n_iter, X_train, y_train, X_test, y_test):
    print(f"Starting logistic regression")
    print(f"Number of Features {features} | Learning Rate {mu} | Iterations {lr_n_iter}")
    start_time = time.time()
    lrgd = logisticRegression.LogisticRegressionGD(n_iter=lr_n_iter, eta=mu)
    lrgd.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = lrgd.predict(X_test)
    # Evaluate accuracy
    acc = (y_pred == y_test.int()).float().mean()
    total_time = time.time() - start_time
    print(f"\n Training completed in {total_time / 60:.2f} minutes")
    print(f"Logistic Regression Accuracy: {acc:.4f}")

#FNN from previous section moved to its own function.
def fnn(review_data, tfidf_testing_norm,y_test, num_features, mu, L2_reg):
    start_time = time.time()
    nf = int(num_features)
    nf_2 = int(num_features/2)
    nf_4 = int(num_features/4)

    print("\n4. Initializing neural network...")
    net = torch.nn.Sequential(
        torch.nn.Linear(nf, nf_2),
        torch.nn.ReLU(),
        torch.nn.Linear(nf_2, nf_2),
        torch.nn.ReLU(),
        torch.nn.Linear(nf_2, nf_2),
        torch.nn.ReLU(),
        torch.nn.Linear(nf_2, nf_4),
        torch.nn.ReLU(),
        torch.nn.Linear(nf_4, 2),
        torch.nn.Softmax(dim=1))

    optimizer = torch.optim.SGD(net.parameters(), lr=mu, weight_decay=L2_reg)
    L = torch.nn.CrossEntropyLoss()

    # Train. It's doing something and it takes a while. I got to take a shower and it barely finished the
    # Second epoch. So to finish 5 epochs, likely 1 hour and 30 minutes.
    print(f"Number of Features {num_features} | Learning Rate {mu} | L2 Regularization {L2_reg}")
    print("\n5. starting FNN training...")
    print("Epoch | Training Loss | Test Accuracy | Time")
    print("-" * 40)

    for epoch in range(10):
        epoch_start = time.time()
        net.train()
        total_loss = 0
        batch_count = 0

        for (x, y) in review_data:
            output = net.forward(x.view(-1, nf))
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
            # print(acc)
            epoch_time = time.time() - epoch_start
            print(f"{epoch + 1:5d} | {avg_loss:.6f} | {acc:.6f} | {epoch_time:.2f}s")

    total_time = time.time() - start_time
    print(f"\n Training completed in {total_time / 60:.2f} minutes")
    print(f"Final FNN test accuracy: {acc:.4f}")

if __name__ == "__main__":
    process_data()