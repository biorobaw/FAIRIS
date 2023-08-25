import math
import random
import xml.etree.ElementTree as ET
from random import uniform

import numpy as np
import pandas as pd


def allXall(df1, df2):
    return df1.merge(df2, on='key', sort=False)


def dataFrame(colname, values):
    if isinstance(values, list) or isinstance(values, np.ndarray):
        # print(values)
        return pd.DataFrame({'key': 0, colname: values})
    else:
        return pd.DataFrame({'key': 0, colname: [values]})


def load_maze_random_default(number_obstacles=10, file_name='outmaze.xml'):
    # maze for bio experiments:
    #		obstacle length = 0.19*sqrt(0.3538) = 0.113 # biology length * distance scale ratio
    # 		num obstacles: 0, 6, 11, 23
    # maze for robot experiments:
    #		obstacle length 25cm
    #		num obstacles: 0, 10, 20, 30, 40, 50, 60
    # NOTE: min distance between obstacles set to 10cm
    walls = pd.concat([external_walls(), default_random_obstacle(number_obstacles)], ignore_index=True)
    goal = goal_maze_default()
    experiment_starts = experiment_start_pos_maze_default()
    habituation_starts = habituation_start_pos_maze_default()
    write_maze_file(walls, goal, experiment_starts, habituation_starts, file_name)


def external_walls():
    return pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'],
                        data=[
                            [-3.0, -3.0, -3.0, 3.0],
                            [-3.0, 3.0, 3.0, 3.0],
                            [3.0, -3.0, 3.0, 3.0],
                            [-3.0, -3.0, 3.0, -3.0]
                        ]
                        )


def dummy_walls():
    return pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])


def goal_maze_default():
    return pd.DataFrame(
        columns=['id', 'x', 'y'],
        data=[
            [int(1), 0.0, 0.0]
        ]
    )


def experiment_start_pos_maze_default():
    return pd.DataFrame(
        columns=['x', 'y', 'theta'],
        data=[
            [0.0, 2.5, math.pi/2],
            [0.0, -2.5, math.pi/2],
            [-2.5, 0.0, math.pi/2],
            [2.5, 0.0, math.pi/2]
        ]
    )


def habituation_start_pos_maze_default():
    return pd.DataFrame(
        columns=['x', 'y', 'theta'],
        data=[
            [-2.5, -2.5, math.pi/2],
            [2.5, -2.5, math.pi/2],
            [-2.5, 2.5, math.pi/2],
            [2.5, 2.5, math.pi/2]
        ]
    )


def make_obstacle(x, y):
    obstacle_length = .5
    theta = uniform(0, 2 * math.pi)
    x1 = x + (obstacle_length / 2) * (math.cos(theta))
    y1 = y + (obstacle_length / 2) * (math.sin(theta))
    x2 = x + (obstacle_length / 2) * (math.cos(theta + math.pi))
    y2 = y + (obstacle_length / 2) * (math.sin(theta + math.pi))
    return [x1, y1, x2, y2]


def default_random_obstacle(number_obstacle):
    x_centers = [-2.5, -1.5, -.5, .5, 1.5, 2.5]
    y_centers = [-2.5, -1.5, -.5, .5, 1.5, 2.5]
    not_valid_positions = [(-2.5, 2.5),
                           (-2.5, -2.5),
                           (2.5, -2.5),
                           (2.5, 2.5),
                           (0.5, 2.5),
                           (-0.5, 2.5),
                           (0.5, -2.5),
                           (-0.5, -2.5),
                           (2.5, -0.5),
                           (-2.5, -0.5),
                           (2.5, 0.5),
                           (-2.5, 0.5),
                           (0.5, 0.5),
                           (-0.5, -0.5),
                           (0.5, -0.5),
                           (-0.5, 0.5)]
    possible_positions = []
    for x in x_centers:
        for y in y_centers:
            if not ((x, y) in not_valid_positions):
                possible_positions.append((x, y))

    random_positions = random.sample(possible_positions, number_obstacle)
    obstacles = []
    for p in random_positions:
        obstacles.append(make_obstacle(p[0], p[1]))

    return pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'],
                        data=obstacles
                        )


def write_maze_file(walls, goals, experiment_starts, habituation_starts, file_name):
    world = ET.Element('world')
    experiment_starts_xml = ET.SubElement(world, 'experimentStartPositions')
    for index, row in experiment_starts.iterrows():
        pos = ET.SubElement(experiment_starts_xml, 'pos', x=str(row['x']), y=str(row['y']), theta=str(row['theta']))

    habituation_starts_xml = ET.SubElement(world, 'habituationStartPositions')
    for index, row in habituation_starts.iterrows():
        pos = ET.SubElement(habituation_starts_xml, 'pos', x=str(row['x']), y=str(row['y']), theta=str(row['theta']))

    for index, row in goals.iterrows():
        goal = ET.SubElement(world, 'goal', id=str(int(row['id'])), x=str(row['x']), y=str(row['y']))
    for index, row in walls.iterrows():
        wall = ET.SubElement(world, 'wall', x1=str(row['x1']), y1=str(row['y1']), x2=str(row['x2']), y2=str(row['y2']))
    tree = ET.ElementTree(world)
    ET.indent(tree, space="\t", level=0)
    tree.write(file_name, xml_declaration=True)

