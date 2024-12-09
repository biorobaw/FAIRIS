import os
import torch
os.chdir("../../..")
print(os.getcwd())
from fairis_lib.robot_lib.rosbot import RosBot
from fairis_tools.experiment_tools.loggers.visual_data_set import PovDataset
import pickle

maze_file_dir = 'simulation/worlds/mazes/VisualPlaceCells/'

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
number_steps = 250
# create the robot/supervisor instance.
robot = RosBot(enable_cnn_features=True)

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
            robot_x, robot_y, robot_theta = robot.get_robot_pose()
            multimodal_features, cnn_features, landmark_mask = robot.get_robot_pov_features(landmark_dictionary)

            print(robot_x, robot_y, robot_theta)

            dataset.add_observations(multimodal_features, cnn_features, robot_x, robot_y, robot_theta, landmark_mask)

    dataset.save_dataset("data/VisualPlaceCellData/LM8_1000")
robot.experiment_supervisor.simulationReset()
