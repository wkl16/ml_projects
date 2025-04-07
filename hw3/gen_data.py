import idx2numpy
import os
import matplotlib.pyplot as plt
import numpy as np

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