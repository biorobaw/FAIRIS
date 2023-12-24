import os
import time
from datetime import timedelta
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

"""ExperimentSupervisor controller."""
from BiologyTools.PlaceCellLibrary import PlaceCellNetwork
from Simulation.libraries.RosBot import RosBot

from tensorflow import keras

os.chdir("../../")

PC_VALUE_AT_RADIUS = 0.6

directions = 6

if directions == 2:
    model_name = 'controllers/Example/2_directions.h5'
else:
    model_name = 'controllers/Example/6_directions.h5'


def __calc_activation_matrix__(path, pcs, localization_noise=0, percentual_noise=0, additive_noise=0):
    """ Calculate a matrix containing the activation of all place cells for all times.
        Each row represents a place cell, while columns represent the time index.
        Both 'pos' and 'pcs' are data frames containing the path and the set of place cells.
    """
    # get number of pcs and position in path
    num_pcs = len(pcs)
    num_pos = len(path)

    # convert data to numpy to operate
    radii = pcs['placeradius'].to_numpy()
    pcs = pcs[['x', 'y']].to_numpy()
    pos = path[['x', 'y']].to_numpy()

    # replicate the position vector by the number of place cells for easy operations
    pos_tile = pos.reshape(1, -1, 2)
    pos_all = np.tile(pos_tile, (num_pcs, 1, 1))

    # replicate the place cells and radii by the number of positions for easy operations
    pcs_tile = pcs.reshape(-1, 1, 2)
    pcs_all = np.tile(pcs_tile, (1, num_pos, 1))
    radii_all = np.tile(radii.reshape((-1, 1)), (1, num_pos))

    # calculate the activations (see description of formula at the top of this file)
    delta = pos_all - pcs_all
    delta2 = (delta * delta).sum(2)

    if localization_noise != 0:
        d_error = np.random.uniform(1 - localization_noise, 1 + localization_noise, delta2.shape)
        delta2 *= d_error * d_error

    r2 = radii_all * radii_all
    exponents = np.log(PC_VALUE_AT_RADIUS) * delta2 / r2
    activations = np.exp(exponents)

    if percentual_noise != 0:
        activations *= np.random.uniform(1 - percentual_noise, 1 + percentual_noise, activations.shape)

    if additive_noise != 0:
        activations = np.minimum(
            np.maximum(0, activations + np.random.uniform(-additive_noise, additive_noise, activations.shape)), 1)

    return activations


maze_file = 'worlds/mazes/Samples/square.xml'

# create the robot/supervisor instance.
robot = RosBot()

# Loads the environment from the maze file
robot.load_environment(maze_file)

# Show basic robot/supervisor functions
# robot.move_to_random_start()

pcs = pd.read_csv('controllers/Example/placecells.csv', delim_whitespace=True)

# Creates Place Cell Network
experiment_pc_network = PlaceCellNetwork()

# for i in range(len(pcs)):
#     experiment_pc_network.add_pc_to_network(
#         robot_x=pcs[i][0] * 4.5,
#         robot_y=pcs[i][0] * 4.5,
#         radius=radii[i] * 4.5
#     )

SIZE_FACTOR = 4.5

FAST = robot.max_motor_velocity
MEDIUM = robot.max_motor_velocity / 2
SLOW = robot.max_motor_velocity / 3


# S-F-S [0] - 90º
# F-S-S [1] - 270º

# FAST 3-6
# MEDIUM 7-10
# SLOW 11-14

# S-F-S [0] - 90º
# F-S-S [1] - 270º

# S-F-M-F [0] - 90º
# F-S-M-F [1] - 270º
# M-F-S-F [2] - 225º
# M-S-F-F [3] - 135º
# S-M-F-F [4] - 180º
# F-M-S-F [5] - 45º

def split_segment(x1, y1, x2, y2, N):
    points = []
    for i in range(N + 1):
        ratio = i / N  # Calculate the ratio of the current point
        new_x = x1 + ratio * (x2 - x1)  # Calculate the x-coordinate of the new point
        new_y = y1 + ratio * (y2 - y1)  # Calculate the y-coordinate of the new point
        points.append((new_x, new_y))
    return points


def getEncodingFromAngle(angle):
    if directions == 6:
        match angle:
            case 90:
                print('SLOW, FAST, MEDIUM, FAST')
                return SLOW, FAST, MEDIUM, FAST
            case 270:
                print('FAST, SLOW, MEDIUM, FAST')
                return FAST, SLOW, MEDIUM, FAST
            case 0:
                print('MEDIUM, FAST, SLOW, FAST')
                return MEDIUM, FAST, SLOW, FAST
            case 135:
                print('MEDIUM, SLOW, FAST, FAST')
                return MEDIUM, SLOW, FAST, FAST
            case 180:
                print('SLOW, MEDIUM, FAST, FAST')
                return SLOW, MEDIUM, FAST, FAST
            case 45:
                print('FAST, MEDIUM, SLOW, FAST')
                return FAST, MEDIUM, SLOW, FAST
            case default:
                print(f'{angle=} not valid')
                return FAST, FAST, FAST, FAST
    else:
        match angle:
            case 90:
                print('SLOW, FAST, SLOW')
                return SLOW, FAST, SLOW
            case 270:
                print('FAST, SLOW, SLOW')
                return FAST, SLOW, SLOW


def moveRobot(x, y, angle):
    print(f'Should predict {angle=}°')

    current_position = robot.gps.getValues()[0:2]

    if directions == 6:
        num_segments = 4
    else:
        num_segments = 3

    points = split_segment(current_position[0], current_position[1], x, y, num_segments)

    velocities = getEncodingFromAngle(angle)

    pos = []
    segments = []
    len_seg = 0

    for i in range(1, len(points)):
        pos1 = robot.forward_motion_to_xy(x=points[i][0], y=points[i][1], velocity=velocities[i - 1])
        # print(f'Number of positions = {len(pos1)}')
        len_seg += len(pos1)
        segments.append(len_seg)
        pos.extend(pos1)

    return pos, segments


def calculateActivations(pos, segments=None):
    pos_pcs = np.array(pos) / SIZE_FACTOR

    path = pd.DataFrame(pos_pcs, columns=['x', 'y'])
    activations = __calc_activation_matrix__(path, pcs,
                                             localization_noise=0,
                                             percentual_noise=0,
                                             additive_noise=0)

    plt.close('all')
    plt.imshow(activations, cmap='viridis', aspect='auto')
    for segment in segments:
        plt.axvline(x=segment - 1, color='red', linestyle='--')
        if segment == segments[-2]:
            break
    plt.colorbar()
    plt.xlabel('Timestep')
    plt.ylabel('Activation value per Place Cell')
    plt.show()

    return activations


def getPredictionFromActivations(activations, model):
    angle = [90, 270, 0, 135, 180, 45]

    X = [np.transpose(activations)]
    X = np.array(X)

    print(f'Getting prediction')
    start_time = time.monotonic()

    batch_y_pred = model.predict(X, verbose=0)

    end_time = time.monotonic()
    executionTime = timedelta(seconds=end_time - start_time)

    print(f"--- {executionTime} to predict ---\n")

    print(f'{batch_y_pred=}')

    output = np.argmax(batch_y_pred)

    return angle[output]


print('\nLoading Model')
start_time = time.monotonic()

model = keras.models.load_model(model_name)

end_time = time.monotonic()
executionTime = timedelta(seconds=end_time - start_time)

print(f"--- {executionTime} to load {model_name} ---\n")

print('\n')

if directions == 2:
    robot.teleport_robot(x=-4.5, y=-3.5)

    positions, segments = moveRobot(x=-0.5, y=-3.5, angle=90)

    activations = calculateActivations(positions, segments)

    prediction = getPredictionFromActivations(activations, model)

    print(f'Turning {prediction}°')
    turnAngle = (robot.get_bearing() + prediction + 360) % 360
    robot.rotate_to(turnAngle, margin_error=1)

    print('\n')

    positions, segments = moveRobot(x=-0.5, y=-0.5, angle=270)

    activations = calculateActivations(positions, segments)

    prediction = getPredictionFromActivations(activations, model)

    print(f'Turning {prediction}°')
    turnAngle = (robot.get_bearing() + prediction + 360) % 360
    robot.rotate_to(turnAngle, margin_error=1)

    print('\n')

elif directions == 6:
    robot.teleport_robot(x=-4.5, y=-4.5)

    positions, segments = moveRobot(x=-0.5, y=-4.5, angle=45)

    activations = calculateActivations(positions, segments)

    prediction = getPredictionFromActivations(activations, model)

    print(f'Turning {prediction}°')
    turnAngle = (robot.get_bearing() + prediction + 360) % 360
    robot.rotate_to(turnAngle, margin_error=1)

    print('\n')

    positions, segments = moveRobot(x=4.5, y=0.5, angle=135)

    activations = calculateActivations(positions, segments)

    prediction = getPredictionFromActivations(activations, model)

    print(f'Turning {prediction}°')
    turnAngle = (robot.get_bearing() + prediction + 360) % 360
    robot.rotate_to(turnAngle, margin_error=1)

    print('\n')

    positions, segments = moveRobot(x=0.5, y=0.5, angle=270)

    activations = calculateActivations(positions, segments)

    prediction = getPredictionFromActivations(activations, model)

    print(f'Turning {prediction}°')
    turnAngle = (robot.get_bearing() + prediction + 360) % 360
    robot.rotate_to(turnAngle, margin_error=1)

    print('\n')

    positions, segments = moveRobot(x=0.5, y=3.5, angle=90)

    activations = calculateActivations(positions, segments)

    prediction = getPredictionFromActivations(activations, model)

    print(f'Turning {prediction}°')
    turnAngle = (robot.get_bearing() + prediction + 360) % 360
    robot.rotate_to(turnAngle, margin_error=1)

    print('\n')

    positions, segments = moveRobot(x=-3.5, y=3.5, angle=180)

    activations = calculateActivations(positions, segments)

    prediction = getPredictionFromActivations(activations, model)

    print(f'Turning {prediction}°')
    turnAngle = (robot.get_bearing() + prediction + 360) % 360
    robot.rotate_to(turnAngle, margin_error=1)

    print('\n')

    positions, segments = moveRobot(x=0, y=3.5, angle=0)

    activations = calculateActivations(positions, segments)

    prediction = getPredictionFromActivations(activations, model)

    print(f'Turning {prediction}°')
    turnAngle = (robot.get_bearing() + prediction + 360) % 360
    robot.rotate_to(turnAngle, margin_error=1)

robot.experiment_supervisor.simulationReset()
