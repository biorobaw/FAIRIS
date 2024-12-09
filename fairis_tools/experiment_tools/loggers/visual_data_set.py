import pickle


# Define the Datapoint class with automatic image processing
class NavigationDataPoint:
    def __init__(self, multimodal_feature_vector, cnn_feature_vector, x, y, theta, landmark_mask = [0,0,0,0,0,0,0,0,0]):
        # Process the image and convert it to a tensor
        self.multimodal_feature_vector = multimodal_feature_vector
        self.cnn_feature_vector = cnn_feature_vector

        # Store additional attributes
        self.x = x
        self.y = y
        self.theta = theta
        self.landmark_mask = landmark_mask

# Define the Dataset class
class PovDataset:
    def __init__(self):
        self.observations = []

    def add_observations(self, multimodal_feature_vector, cnn_feature_vector, x, y, theta, landmark_mask):
        """
        Adds a new datapoint to the dataset.
        """
        self.observations.append(NavigationDataPoint(multimodal_feature_vector, cnn_feature_vector, x, y, theta, landmark_mask))

    def save_dataset(self, filename):
        """
        Save the dataset to a file using pickle.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
        print(f"Dataset saved to {filename}")
