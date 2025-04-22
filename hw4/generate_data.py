import pyprind
import pandas as pd
import numpy as np
import os
import sys
import re
import time
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
        labels = {'pos':1, 'neg':0}
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
        df.columns = ['review',  'sentiment']
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
    
    for epoch in range(5):
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
            # print(acc)
            epoch_time = time.time() - epoch_start
            print(f"{epoch+1:5d} | {avg_loss:.6f} | {acc:.6f} | {epoch_time:.2f}s")

    total_time = time.time() - start_time
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


if __name__ == "__main__":
    process_data()
