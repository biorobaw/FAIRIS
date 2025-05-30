import os
import torch
os.chdir("../../..")
print(os.getcwd())
from fairis_lib.robot_lib.hambot import HamBot
from fairis_tools.experiment_tools.loggers.visual_data_set import PovDataset
import json
import random

maze_file_dir = 'simulation/worlds/mazes/VisualPlaceCells/'
path_point_dir = 'data/PathPoints/VPC_Walls.json'

with open(path_point_dir,"r") as file:
    pose_list = json.load(file)

# Ensure reproducibility
# random.seed(42)
#
# # Shuffle the pose list
# random.shuffle(pose_list)

# Define the split ratio
# split_ratio = 0.8  # 80% for training, 20% for testing
#
# # Determine the split index
# split_index = int(len(pose_list) * split_ratio)
#
# # Split the list into training and testing
# train_list = pose_list[:split_index]
# test_list = pose_list[split_index:]


landmark_dictionary = {
    0: [1.0, 0.0, 0.0],
    1: [0.0, 1.0, 0.0],
    2: [0.0, 0.0, 1.0],
    3: [1.0, 1.0, 0.0],
    4: [1.0, 0.0, 1.0],
    5: [0.0, 1.0, 1.0],
    6: [1.0, 0.5, 0.0],
    7: [0.5, 0.0, 0.5],
    8: [1.0, 0.2, 1.0],
}

maze_files = ['LM4','LM6','LM8','LMO8','LM8_addition','LMO8_remove']
maze_index = 3
# create the robot/supervisor instance.
robot = HamBot(enable_cnn_features=False)
robot.load_environment(maze_file_dir + maze_files[maze_index] + '.xml')
training_dataset = PovDataset()

thetas = [0.0, 0.7854, 1.5708, 2.3562, 3.1416, 3.9270, 4.7124, 5.4978]

for start_position in robot.maze.experiment_starting_location:

    robot.teleport_robot(x=start_position.x, y=start_position.y, theta=start_position.theta)
    for i in range(200):
        action = robot.perform_habituation_action()
        robot_x, robot_y, robot_theta = robot.get_robot_pose()
        print("Random Action: ", i)
        for theta in thetas:
            robot.teleport_robot(x=robot_x,y=robot_y,theta=theta)
            robot_x, robot_y, robot_theta = robot.get_robot_pose()
            multimodal_features, cnn_features, landmark_mask = robot.get_robot_pov_features(landmark_dictionary)

            # print(robot_x, robot_y, robot_theta)

            training_dataset.add_observations(multimodal_features, cnn_features, robot_x, robot_y, robot_theta,
                                          landmark_mask)

# training_dataset.save_dataset("data/VisualPlaceCellData/"+maze_files[maze_index]+"_Training")

# for pose in pose_list:
#     print("Random Action: ", pose)
#     x = pose[0]
#     y = pose[1]
#     theta = pose[2]
#     robot.teleport_robot(x=x, y=y, theta=theta)
#     robot_x, robot_y, robot_theta = robot.get_robot_pose()
#     multimodal_features, cnn_features, landmark_mask = robot.get_robot_pov_features(landmark_dictionary)
#
#     print(robot_x, robot_y, robot_theta)
#
#     training_dataset.add_observations(multimodal_features, cnn_features, x, y, theta, landmark_mask)
#
# training_dataset.save_dataset("data/VisualPlaceCellData/"+maze_files[maze_index]+"_base_Testing_Wall")


# for pose in train_list:
#     print("Random Action: ", pose)
#     x = pose[0]
#     y = pose[1]
#     theta = pose[2]
#     robot.teleport_robot(x=x, y=y, theta=theta)
#     robot_x, robot_y, robot_theta = robot.get_robot_pose()
#     multimodal_features, cnn_features, landmark_mask = robot.get_robot_pov_features(landmark_dictionary)
#
#     print(robot_x, robot_y, robot_theta)
#
#     training_dataset.add_observations(multimodal_features, cnn_features, robot_x, robot_y, robot_theta, landmark_mask)
#
# training_dataset.save_dataset("data/VisualPlaceCellData/"+maze_files[maze_index]+"_Training")

# testing_dataset = PovDataset()
#
# for pose in test_list:
#     print("Random Action: ", pose)
#     x = pose[0]
#     y = pose[1]
#     theta = pose[2]
#     robot.teleport_robot(x=x, y=y, theta=theta)
#     robot_x, robot_y, robot_theta = robot.get_robot_pose()
#     multimodal_features, cnn_features, landmark_mask = robot.get_robot_pov_features(landmark_dictionary)
#
#     print(robot_x, robot_y, robot_theta)
#
#     testing_dataset.add_observations(multimodal_features, cnn_features, robot_x, robot_y, robot_theta, landmark_mask)
#
# testing_dataset.save_dataset("data/VisualPlaceCellData/"+maze_files[maze_index]+"Testing")
robot.experiment_supervisor.simulationReset()

