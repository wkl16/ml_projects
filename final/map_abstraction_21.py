import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Map_Abstraction:
    def __init__(self):
        self.X = None

    def abstract(self, image_path, x_size, y_size):
        image = Image.open(image_path).convert("1")
        image_array = np.array(image)
        # black = 1, white = 0
        binary_array = (image_array == 0).astype(int)
        h, w = binary_array.shape
        output = np.zeros((y_size, x_size), dtype=int)
        for i in range(y_size):
            y_start = int(i * h / y_size)
            y_end = int((i + 1) * h / y_size)
            for j in range(x_size):
                x_start = int(j * w / x_size)
                x_end = int((j + 1) * w / x_size)
                block = binary_array[y_start:y_end, x_start:x_end]
                output[i, j] = int(block.max())

        return output

    def show_plot(self, matrix, show_grid=False):       
        fig = plt.subplots(figsize=(8, 8))
        fig.imshow(matrix, cmap='gray_r', interpolation='nearest', vmin=0, vmax=1)

        if show_grid:
            y_size, x_size = matrix.shape
            fig.set_xticks(np.arange(-0.5, x_size, 1), minor=True)
            fig.set_yticks(np.arange(-0.5, y_size, 1), minor=True)
            fig.grid(which='minor', color='gray', linewidth=0.5)
            fig.tick_params(which='minor', size=0) 

        plt.show()
