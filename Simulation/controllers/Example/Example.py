import os
os.chdir("../../..")

"""ExperimentSupervisor controller."""
from BiologyTools.PlaceCellLibrary import PlaceCellNetwork
from Simulation.libraries.RobotLib.RosBot import RosBot


maze_file = 'Simulation/worlds/mazes/Samples/LandmarkExample.xml'

# create the robot/supervisor instance.
robot = RosBot()

# Loads the environment from the maze file
robot.load_environment(maze_file)

# Show basic robot/supervisor functions
robot.move_to_random_experiment_start()

# Creates Place Cell Network
experiment_pc_network = PlaceCellNetwork()

for i in range(8):
    robot.perform_action_with_PID(i)
    robot_x, robot_y, robot_theta = robot.get_robot_pose()
    if experiment_pc_network.get_num_active_pc(robot_x,robot_y) == 0:
        experiment_pc_network.add_pc_to_network(robot_x,robot_y,)
    experiment_pc_network.activate_pc_network(robot_x, robot_y)
    # experiment_pc_network.print_pc_activations()
    robot.update_pc_display(experiment_pc_network.pc_list[-1])


robot.experiment_supervisor.simulationReset()


