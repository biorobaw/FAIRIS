import os
import torch
os.chdir("../../..")
print(os.getcwd())
from fairis_lib.robot_lib.rosbot import RosBot
from fairis_tools.experiment_tools.loggers.visual_data_set import PovDataset
import json
import random

maze_file_dir = 'simulation/worlds/mazes/VisualPlaceCells/'
path_point_dir = 'data/PathPoints/VPC1.json'

with open(path_point_dir,"r") as file:
    pose_list = json.load(file)

# Ensure reproducibility
random.seed(42)

# Shuffle the pose list
random.shuffle(pose_list)

# Define the split ratio
split_ratio = 0.8  # 80% for training, 20% for testing

# Determine the split index
split_index = int(len(pose_list) * split_ratio)

# Split the list into training and testing
train_list = pose_list[:split_index]
test_list = pose_list[split_index:]


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

maze_files = ['LM8']
# create the robot/supervisor instance.
robot = RosBot(enable_cnn_features=True)
robot.load_environment(maze_file_dir + maze_files[0] + '.xml')
training_dataset = PovDataset()

for pose in train_list:
    print("Random Action: ", pose)
    x = pose[0]
    y = pose[1]
    theta = pose[2]
    robot.teleport_robot(x=x, y=y, theta=theta)
    robot_x, robot_y, robot_theta = robot.get_robot_pose()
    multimodal_features, cnn_features, landmark_mask = robot.get_robot_pov_features(landmark_dictionary)

    print(robot_x, robot_y, robot_theta)

    training_dataset.add_observations(multimodal_features, cnn_features, robot_x, robot_y, robot_theta, landmark_mask)

training_dataset.save_dataset("data/VisualPlaceCellData/LM8_Training")

testing_dataset = PovDataset()

for pose in test_list:
    print("Random Action: ", pose)
    x = pose[0]
    y = pose[1]
    theta = pose[2]
    robot.teleport_robot(x=x, y=y, theta=theta)
    robot_x, robot_y, robot_theta = robot.get_robot_pose()
    multimodal_features, cnn_features, landmark_mask = robot.get_robot_pov_features(landmark_dictionary)

    print(robot_x, robot_y, robot_theta)

    testing_dataset.add_observations(multimodal_features, cnn_features, robot_x, robot_y, robot_theta, landmark_mask)

testing_dataset.save_dataset("data/VisualPlaceCellData/LM8_Testing")
robot.experiment_supervisor.simulationReset()

