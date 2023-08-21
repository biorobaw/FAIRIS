import math
from random import *
from matplotlib import collections as pycol
import matplotlib.pyplot as plt

from Simulation.libraries.MazeAndPcsParcer import parse_maze


class Maze:
    def __init__(self, maze_file):

        self.length = 0
        self.width = 0
        self.boundary_walls = []
        self.starting_location = []
        self.goal_locations = []
        self.obstacle = []
        self.walls = []

        walls, goals, start_positions = parse_maze(maze_file)

        for index, row in walls.iterrows():
            if index <= 3:
                self.boundary_walls.append(BoundryWall(row['x1'], row['y1'], row['x2'], row['y2'], id=index))
            else:
                self.obstacle.append(Obstacle(row['x1'], row['y1'], row['x2'], row['y2'], id=index - 4))
            self.walls.append([(row['x1'], row['y1']), (row['x2'], row['y2'])])

        for index, row in start_positions.iterrows():
            self.starting_location.append(StartingPosition(row['x'], row['y']))

        for index, row in goals.iterrows():
            self.goal_locations.append(Goal(row['x'], row['y'], row['id']))

    # Returns random starting positions
    def get_random_starting_position(self):
        return sample(self.starting_location, 1)[0]

    # Creates a matplotlib plot of the maze
    def get_maze_figure(self,display_width,display_height):
        self.maze_figure, self.maze_figure_ax = plt.subplots(figsize=(display_width/100,display_height/100))
        self.maze_figure_ax.add_collection(pycol.LineCollection(self.walls,linewidths=2))
        self.maze_figure_ax.set_ylim(-3, 3)
        self.maze_figure_ax.set_xlim(-2, 2)
        self.maze_figure_ax.margins(0.1)
        return self.maze_figure, self.maze_figure_ax



class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance_to_point(self, robot_position):
        return math.dist((self.x, self.y), (robot_position[0], robot_position[1])) - robot_position[2]


class Goal(Point):
    def __init__(self, x, y, id):
        super().__init__(x, y)
        self.goal_id = id


class StartingPosition(Point):
    def __init__(self, x, y):
        super().__init__(x, y)


class BoundryWall:
    def __init__(self, x1, y1, x2, y2, height=0.5, width=0.012, id=0):
        self.end_point1 = Point(x1, y1)
        self.end_point2 = Point(x2, y2)
        self.height = height
        self.width = width
        self.length = math.dist((x1, y1), (x2, y2))
        self.id = id
        self.center_mass = Point((self.end_point1.x + self.end_point2.x) / 2,
                                 (self.end_point1.y + self.end_point2.y) / 2)
        self.dimensions = [self.width, self.length, self.height]
        self.translation = [(self.end_point1.x + self.end_point2.x) / 2,
                            (self.end_point1.y + self.end_point2.y) / 2,
                            self.height / 2]
        theta = math.atan2(self.end_point1.x - self.end_point2.x, self.end_point1.y - self.end_point2.y)
        self.rotation = [0, 0, 1, theta]

    def get_webots_translation_string(self):
        txt = 'translation {x:.2f} {y:.2f} {z:.2f}'
        return txt.format(x=self.translation[0], y=self.translation[1], z=self.translation[2])

    def get_webots_rotation_string(self):
        txt = 'rotation {x:.2f} {y:.2f} {z:.2f} {theta:-.2f}'
        return txt.format(x=self.rotation[0], y=self.rotation[1], z=self.rotation[2], theta=self.rotation[3])

    def get_webots_size_string(self):
        txt = 'size {width:.2f} {length:.2f} {height:.2f}'
        return txt.format(width=self.width, length=self.length, height=self.height)

    def get_webots_node_string(self):
        node_string = "{translation} {rotation} {size}".format(translation=self.get_webots_translation_string(),
                                                               rotation=self.get_webots_rotation_string(),
                                                               size=self.get_webots_size_string())
        return 'DEF Boundary_Wall{id} Obstacle '.format(id=self.id) + '{ ' + node_string + ' }'


class Obstacle:
    def __init__(self, x1, y1, x2, y2, height=0.5, width=0.012, id=0):
        self.end_point1 = Point(x1, y1)
        self.end_point2 = Point(x2, y2)
        self.height = height
        self.width = width
        self.length = math.dist((x1, y1), (x2, y2))
        self.id = id
        self.center_mass = Point((self.end_point1.x + self.end_point2.x) / 2,
                                 (self.end_point1.y + self.end_point2.y) / 2)
        self.dimensions = [self.width, self.length, self.height]
        self.translation = [(self.end_point1.x + self.end_point2.x) / 2,
                            (self.end_point1.y + self.end_point2.y) / 2,
                            self.height / 2]
        theta = math.atan2(self.end_point1.x - self.end_point2.x, self.end_point1.y - self.end_point2.y)
        self.rotation = [0, 0, 1, math.pi - theta]

    def get_webots_translation_string(self):
        txt = 'translation {x:.2f} {y:.2f} {z:.2f}'
        return txt.format(x=self.translation[0], y=self.translation[1], z=self.translation[2])

    def get_webots_rotation_string(self):
        txt = 'rotation {x:.2f} {y:.2f} {z:.2f} {theta:-.2f}'
        return txt.format(x=self.rotation[0], y=self.rotation[1], z=self.rotation[2], theta=self.rotation[3])

    def get_webots_size_string(self):
        txt = 'size {width:.2f} {length:.2f} {height:.2f}'
        return txt.format(width=self.width, length=self.length, height=self.height)

    def get_webots_node_string(self):
        node_string = "{translation} {rotation} {size}".format(translation=self.get_webots_translation_string(),
                                                               rotation=self.get_webots_rotation_string(),
                                                               size=self.get_webots_size_string())
        return 'DEF Obstacle_{id} Obstacle '.format(id=self.id) + '{ ' + node_string + ' }'
