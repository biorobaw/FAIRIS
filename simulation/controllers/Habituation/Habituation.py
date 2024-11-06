import os
import pickle

os.chdir("../..")

"""ExperimentSupervisor controller."""
from fairis_tools.biology_tools import PlaceCellNetwork
from fairis_lib.robot_lib import rosbot

mazes_dir = 'simulation/worlds/mazes/Experiment1/'
place_cell_gen_threshold = [1e-2,1e-1,1]
mazes = ['WM00.xml','WM10.xml','WM20.xml']
# create the robot/supervisor instance.
robot = RosBot()

for maze in mazes:
    for run_id in range(5):
        threshold_index = 0
        for threshold in place_cell_gen_threshold:
            maze_file = mazes_dir + maze

            pc_network_name = maze_file.split('/')[-1].split('.')[0] + '_' + str(threshold_index) + '_' + str(run_id)
            threshold_index += 1

            # Loads the environment from the maze file
            robot.load_environment(maze_file)

            # Creates Place Cell Network
            experiment_pc_network = PlaceCellNetwork()
            robot.update_pc_display(experiment_pc_network.pc_list[-1])

            # Preforms Habituation and creates the place cell network
            for habituation_position in range(len(robot.maze.habituation_start_location)):
                # Show basic robot/supervisor functions
                robot.move_to_habituation_start(index=habituation_position)

                for i in range(200):
                    print("Random Action: ", i)
                    robot.perform_random_action()

                    robot_x, robot_y, robot_theta = robot.get_robot_pose()
                    experiment_pc_network.activate_pc_network(robot_x, robot_y)

                    if experiment_pc_network.get_total_pc_activation() < threshold:
                        experiment_pc_network.add_pc_to_network(robot_x, robot_y, radius=robot.get_min_lidar_reading())
                        robot.update_pc_display(experiment_pc_network.pc_list[-1])

            with open("simulation/GeneratedPCNetworks/" + pc_network_name, 'wb') as pc_file:
                pickle.dump(experiment_pc_network, pc_file)

            robot.reset_environment()

robot.experiment_supervisor.simulationReset()