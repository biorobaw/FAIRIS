"""ExperimentSupervisor controller."""

from Simulation.libraries.RosBot import RosBot


maze_file = 'worlds/mazes/experiment2/WM10.xml'

# create the robot/supervisor instance.
robot = RosBot()

# Loads the environment from the maze file
robot.load_environment(maze_file)

# Show basic robot/supervisor functions
robot.move_to_random_start()

robot.move_to_xy(1.0,1.0)

robot.experiment_supervisor.simulationReset()


