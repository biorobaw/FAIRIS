"""ExperimentSupervisor controller."""
from BiologyTools.PlaceCellLibrary import PlaceCellNetwork
from Simulation.libraries.RosBot import RosBot


maze_file = 'worlds/mazes/experiment2/WM00.xml'

# create the robot/supervisor instance.
robot = RosBot()

# Loads the environment from the maze file
robot.load_environment(maze_file)

# Show basic robot/supervisor functions
robot.move_to_random_start()

# Creates Place Cell Network
experiment_pc_network = PlaceCellNetwork()

for i in range(5):
    robot.preform_random_action()
    robot_x, robot_y, robot_theta = robot.get_robot_pose()
    if experiment_pc_network.get_num_active_pc(robot_x,robot_y) == 0:
        experiment_pc_network.add_pc_to_network(robot_x,robot_y,)
    experiment_pc_network.calculate_total_pc_activation(robot_x, robot_y)
    experiment_pc_network.print_pc_activations()
    robot.update_pc_display()

#
# robot.move_to_xy(1.0,1.0)

# robot.experiment_supervisor.simulationReset()


