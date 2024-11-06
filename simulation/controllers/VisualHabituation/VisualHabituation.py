import math
import os
import numpy as np
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
import torch
from PIL import Image
os.chdir("../../..")
print(os.getcwd())

from fairis_lib.robot_lib.rosbot import RosBot
from fairis_tools.experiment_tools.loggers.visual_data_set import PovDataset

maze_file_dir = 'simulation/worlds/mazes/VisualPlaceCells/'

# Define the preprocessing transformation globally
preprocess = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])


# Define the Datapoint class with automatic image processing
class Datapoint:
    def __init__(self, raw_image_data, additional_info=None):
        """
        raw_image_data: The raw image data in list<list<list<int>>> format (e.g., from Webots camera).
        additional_info: Dictionary or other data structure to store additional attributes (e.g., robot pose, time).
        """
        # Process the image and convert it to a tensor
        self.image_tensor = self.process_image(raw_image_data)

        # Store additional attributes
        self.additional_info = additional_info if additional_info is not None else {}

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

landmark_dictionary = {
    0: [1.0, 0.0, 0.0],
    1: [0.0, 1.0, 0.0],
    2: [0.0, 0.0, 1.0],
    3: [1.0, 1.0, 0.0],
    4: [1.0, 0.0, 1.0],
    5: [0.0, 1.0, 1.0],
    6: [1.0, 0.5, 0.0],
    7: [0.5, 0.0, 0.5],
    8: [0.5, 0.5, 0.0],
}


# Load pre-trained ResNet50 model and remove classification layers for feature extraction
resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-2])  # Remove the fully connected layers
resnet50.eval()  # Set model to evaluation mode  # Set model to evaluation mode

maze_files = ['LM8']
number_steps = 500
# create the robot/supervisor instance.
robot = RosBot()

for maze in maze_files:
    robot.load_environment(maze_file_dir + maze + '.xml')
    dataset = PovDataset()
    # Preforms Habituation and creates the place cell network
    for habituation_position in range(len(robot.maze.habituation_start_location)):
        # Show basic robot/supervisor functions
        robot.move_to_habituation_start(index=habituation_position)

        for i in range(number_steps):
            print("Random Action: ", i)
            robot.perform_random_action()
            image_tensor = process_image_for_resnet(robot.get_robot_pov_processed(landmark_dictionary))
            # image_tensor = image_tensor.to('cpu')
            # Extract ResNet50 features
            with torch.no_grad():
                features = resnet50(image_tensor.unsqueeze(0))  # Add batch dimension
                flattened_features = features.view(features.size(0), -1).cpu().numpy().squeeze()

            # Combine landmark mask and orientation
            # landmarks_and_orientation = np.array(landmark_mask + [math.radians(robot.get_bearing())], dtype=np.float32)
            # full_features = np.concatenate((flattened_features, landmarks_and_orientation))
            robot_x, robot_y, robot_theta = robot.get_robot_pose()
            print(robot_x, robot_y, robot_theta)

            # dataset.add_datapoint(raw_image_data=pov, x=robot_x, y=robot_y, theta=math.radians(robot_theta),
            #                       landmark_mask=landmark_mask)

#     dataset.save_dataset("data/VisualPlaceCellData/LM8_1000")
# robot.experiment_supervisor.simulationReset()
