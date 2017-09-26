import image_helpers
import numpy as np
from tqdm import tqdm

class ModelConvolve:
    def train(self, input_images, output_true):
        images_convolved = []
        for image in input_images:
            image_convolved = image_helpers.convolve(image)
            images_convolved.append(image_convolved)
        return np.array(images_convolved)
