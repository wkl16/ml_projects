import idx2numpy
import os
import matplotlib.pyplot as plt
import numpy as np
import zipfile as zf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def generate_data():
    """Generates Necessary Data to be utilized in Assignment 3"""
    # Current Working Directory
    current_dir = os.path.dirname(__file__)
    
    # Specify dataset directory
    # Images MUST BE in this data directory
    data_dir_name = "Image_Data"
    
    # Check for dataset dir. If does not exist, make for preprocessing.
    print("Checking for Dataset Directory...")
    if not os.path.isdir(data_dir_name):
        print("Dataset Directory does not exist, creating directory...")
        os.mkdir(data_dir_name)
        print("Directory successfully created!\n")
    else:
        print("Dataset Directory already exists, moving on...\n")
    
    dataset_dir = os.path.join(current_dir, data_dir_name)

    # Sets up the necessary filepaths and directories for Dataset directory
    data_zip_name = "Image_Data.zip"
    data_zip_path = os.path.join(current_dir, data_zip_name)
    
    # First checks if the dataset directory has items or not
    # I.e. if the zip file has already been extraced into the directory
    empty_check = os.listdir(dataset_dir)
    if not empty_check:
        print("Dataset Directory is empty, populating with necessary data...")
        # Unzips the Dataset and extracts it into the Dataset directory
        with zf.ZipFile(data_zip_path, 'r') as img_zip:
            img_zip.extractall(current_dir)
        print("Data has been populated!\n")
    else:
        print("Dataset has already been populated.\n")


    print("Formatting Data...")
    # Set the n_features, test, and training points
    # for np.reshape
    n_features  = 784
    n_test_pts  = 10000
    n_train_pts = 60000
    
    # Set filepaths for image data
    test_img_path  = os.path.join(dataset_dir, f"t10k-images-idx3-ubyte")
    train_img_path = os.path.join(dataset_dir, f"train-images-idx3-ubyte")
    test_lbl_path  = os.path.join(dataset_dir, f"t10k-labels-idx1-ubyte")
    train_lbl_path = os.path.join(dataset_dir, f"train-labels-idx1-ubyte")
    
    # Convert idx files to numpy arrays
    test_images = idx2numpy.convert_from_file(test_img_path)
    train_images = idx2numpy.convert_from_file(train_img_path)
    test_labels = idx2numpy.convert_from_file(test_lbl_path)
    train_labels = idx2numpy.convert_from_file(train_lbl_path)

    # Flatten testing and training images
    # Labels are already in the shape they need to be in
    flat_test_images = test_images.reshape(n_test_pts, n_features)
    flat_train_images = train_images.reshape(n_train_pts, n_features)

    # Uncomment to see what the image is supposed to look like
    # plt.figure(figsize=(8,5))
    # plt.imshow(test_images[1])
    # plt.show()

    # plt.figure(figsize=(8,5))
    # plt.imshow(train_images[1])
    # plt.show()
    print("Data has been generated and formatted!")
    
    return flat_train_images, flat_test_images, train_labels, test_labels

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()
    
    # print("X_train[0] flattened image:")
    # print(X_train[0])
    # print("\n\n")
    # print("X_test[0] flattened image:")
    # print(X_test[0])