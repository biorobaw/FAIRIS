import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pickle

# Define the preprocessing transformation globally
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])


# Define the Datapoint class with automatic image processing
class Datapoint:
    def __init__(self, raw_image_data, x, y, theta, landmark_mask = [0,0,0,0,0,0,0,0,0]):
        """
        raw_image_data: The raw image data in list<list<list<int>>> format (e.g., from Webots camera).
        additional_info: Dictionary or other data structure to store additional attributes (e.g., robot pose, time).
        """
        # Process the image and convert it to a tensor
        self.image_tensor = self.process_image(raw_image_data)

        # Store additional attributes
        self.x = x
        self.y = y
        self.theta = theta
        self.landmark_mask = landmark_mask

    def process_image(self, raw_image_data):
        """
        Convert raw image data into a tensor, and normalize it using the globally defined preprocess.
        """
        # Convert the raw image data (list<list<list<int>>>) into a NumPy array
        image_array = np.array(raw_image_data, dtype=np.uint8)

        # Convert NumPy array to a PIL Image
        image = Image.fromarray(image_array)

        # Apply the global preprocessing transformation and return the image tensor
        image_tensor = preprocess(image)
        return image_tensor
# Define the Dataset class
class PovDataset:
    def __init__(self):
        self.data = []

    def add_datapoint(self, raw_image_data, x, y, theta, landmark_mask):
        """
        Adds a new datapoint to the dataset.
        """
        self.data.append(Datapoint(raw_image_data, x, y, theta, landmark_mask))

    def save_dataset(self, filename):
        """
        Save the dataset to a file using pickle.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Dataset saved to {filename}")
