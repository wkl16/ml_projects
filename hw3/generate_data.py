import idx2numpy
import os
import matplotlib.pyplot as plt
import numpy as np
import zipfile as zf
import py7zr as un7
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Global Data

# Set the n_features, test, and training points
# for np.reshape
n_features  = 784
n_test_pts  = 10000
n_train_pts = 60000

# Current Working Directory
current_dir = os.path.dirname(__file__)

# Set filepaths for image data
test_img_path  = f"t10k-images-idx3-ubyte"
train_img_path = f"train-images-idx3-ubyte"
test_lbl_path  = f"t10k-labels-idx1-ubyte"
train_lbl_path = f"train-labels-idx1-ubyte"


def generate_data_numbers():
    """Generates Numbers MNIST Data to be utilized in Assignment 3"""
    # Current Working Directory
    current_dir = os.path.dirname(__file__)
    
    # Specify dataset directory
    # Images MUST BE in this data directory
    data_dir_name = "Image_Data_Numbers"
    
    # Check for dataset dir. If does not exist, make for preprocessing.
    print("Checking for Numbers Dataset Directory...")
    if not os.path.isdir(data_dir_name):
        print("Dataset Directory does not exist, creating directory...")
        os.mkdir(data_dir_name)
        print("Directory successfully created!\n")
    else:
        print("Dataset Directory already exists, moving on...\n")
    
    dataset_dir = os.path.join(current_dir, data_dir_name)

    # Sets up the necessary filepaths and directories for Dataset directory
    data_zip_name = "Image_Data_Numbers.zip"
    data_zip_path = os.path.join(current_dir, data_zip_name)
    
    # First checks if the dataset directory has items or not
    # I.e. if the zip file has already been extraced into the directory
    empty_check = os.listdir(dataset_dir)
    if not empty_check:
        print("Dataset Directory is empty, populating with necessary data...")
    else:
        print("Dataset has already been populated/is only partially populated.")
        print("Populating data...")

    # Unzips the Dataset and extracts it into the Dataset directory
    with zf.ZipFile(data_zip_path, 'r') as img_zip:
        img_zip.extractall(current_dir)
    
    print("Data has been populated!\n")

    print("Formatting Data...")

    # Set filepaths for image data
    num_test_img_path  = os.path.join(dataset_dir, test_img_path)
    num_train_img_path = os.path.join(dataset_dir, train_img_path)
    num_test_lbl_path  = os.path.join(dataset_dir, test_lbl_path)
    num_train_lbl_path = os.path.join(dataset_dir, train_lbl_path)
    
    # Convert idx files to numpy arrays
    test_images  = idx2numpy.convert_from_file(num_test_img_path)
    train_images = idx2numpy.convert_from_file(num_train_img_path)
    test_labels  = idx2numpy.convert_from_file(num_test_lbl_path)
    train_labels = idx2numpy.convert_from_file(num_train_lbl_path)

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
    print("Data has been generated and formatted!\n")
    
    return flat_train_images, flat_test_images, train_labels, test_labels

def generate_data_fashion():
    """Generates Fashion MNIST Data to be utilized in Assignment 3"""
    
    # Specify dataset directory
    # Images MUST BE in this data directory
    data_dir_name = "Image_Data_Fashion"
    
    # Check for dataset dir. If does not exist, make for preprocessing.
    print("Checking for Fashion Dataset Directory...")
    if not os.path.isdir(data_dir_name):
        print("Dataset Directory does not exist, creating directory...")
        os.mkdir(data_dir_name)
        print("Directory successfully created!\n")
    else:
        print("Dataset Directory already exists, moving on...\n")
    
    dataset_dir = os.path.join(current_dir, data_dir_name)

    data_set_train_labels_path = os.path.join(dataset_dir, test_img_path)
    data_set_train_images_path = os.path.join(dataset_dir, train_img_path)
    data_set_test_labels_path  = os.path.join(dataset_dir, test_lbl_path) 
    data_set_test_images_path  = os.path.join(dataset_dir, train_lbl_path)

    # Sets up the necessary filepaths and directories for Dataset directory
    data_zip_set_1 = "Fashion_Set_1.zip"
    data_zip_set_2 = "Fashion_Set_2.7z"
    data_zip_set_1_path = os.path.join(current_dir, data_zip_set_1)
    data_zip_set_2_path = os.path.join(current_dir, data_zip_set_2)
    
    # First checks if the dataset directory has items or not
    # I.e. if the zip file has already been extraced into the directory
    empty_check = os.listdir(dataset_dir)
    if not empty_check:
        print("Dataset Directory is empty, populating with necessary data...")
    else:
        print("Dataset has already been populated/is only partially populated.")
        print("Populating data...")

    # Unzips the Dataset and extracts it into the Dataset directory
    with zf.ZipFile(data_zip_set_1_path, 'r') as img_zip:
        img_zip.extractall(dataset_dir)

    with un7.SevenZipFile(data_zip_set_2_path, mode='r') as img_zip:
        img_zip.extractall(dataset_dir)

    print("Data has been populated!\n")

    print("Formatting Data...")
    
    # Convert idx files to numpy arrays
    test_images = idx2numpy.convert_from_file (data_set_train_labels_path)
    train_images = idx2numpy.convert_from_file(data_set_train_images_path)
    test_labels = idx2numpy.convert_from_file (data_set_test_labels_path)
    train_labels = idx2numpy.convert_from_file(data_set_test_images_path)

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
    print("Data has been generated and formatted!\n")
    
    return flat_train_images, flat_test_images, train_labels, test_labels

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data_numbers()
    X_train, X_test, y_train, y_test = generate_data_fashion()
    
    # print("X_train[0] flattened image:")
    # print(X_train[0])
    # print("\n\n")
    # print("X_test[0] flattened image:")
    # print(X_test[0])