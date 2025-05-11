# Task 2.1 - Map abstraction

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Map_Abstraction:
    def abstract(self, image_path, x_size, y_size):
        """Takes a black and white .bmp image and returns compressed numpy matrix of requested x_size and y_size using pooling.
        Parameters
        ------------
        image_path : str
        Path to the .bmp image.
        x_size : int
        Number of columns in output.
        y_size : int
        Number of rows in output.
        Returns
        -------
        2d numpy array : where 0 is open area and 1 reference obstacles in returned matrix.
        """
        #initialize return 2d numpy array.
        abstraction_array = np.zeros((y_size, x_size), dtype=int)

        #Used geeksforgeeks to help convert: https://www.geeksforgeeks.org/how-to-convert-images-to-numpy-array/
        image = Image.open(image_path)
        image_array = np.array(image)

        #Convert numpy image array to where no obstacle (white) is 0 and obstacles (black) are seen as 1 using python
        #bool converted to binary values.
        binary_array = (image_array == 0).astype(int)
        height, width = binary_array.shape

        #Loop to compress converted binary image.
        for i in range(y_size):
            #Grab y range of values for compression based on actual height of image divided by desired size
            #multiplied by current i & i+1 value for range.
            y_start = int(i * height / y_size)
            y_end = int((i + 1) * height / y_size)
            for j in range(x_size):
                #Grab x range of values for compression based on actual width of image divided by desired size
                #multiplied by current i & i+1 value for range.
                x_start = int(j * width / x_size)
                x_end = int((j + 1) * width / x_size)
                #grab actual values from ranges (subsection) of the binary conversion of image.
                subsection = binary_array[y_start:y_end, x_start:x_end]
                #Update output array position, if in that range is an obstacle value return 1 (which is max find a 1)
                #else return 0 for open area.
                abstraction_array[i, j] = int(subsection.max())
        return abstraction_array

    def show_plot(self, matrix, show_grid=False):
        """Takes binary 2d numpy array and prints out plot, used to verify compression of .bmp image.
        Parameters
        ------------
        matrix : 2d numpy array
        This should be an abstraction array.
        show_grid : bool
        Define if the plot should show grid lines, not great for large abstraction arrays.
        Returns
        -------
        none : just prints plot of compressed numpy array
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(matrix, cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)
        if show_grid:
            y_size, x_size = matrix.shape
            plt.xticks(np.arange(-0.5, x_size, 1), minor=True, color='gray')
            plt.yticks(np.arange(-0.5, y_size, 1), [], color='gray')
            plt.grid(which='both', color='gray', linewidth=.5)
        plt.show()
