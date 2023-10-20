import pickle

from ExperimentTools.pc_and_maze_creation.tools.path_planning.wavefront_planner import wave_front_planner, get_path
from Simulation.libraries.SimulationLib.MazeAndPcsParcer import parse_maze_for_wavefront
from Simulation.libraries.SimulationLib.Environment import Maze


class MazePaths:
    def __init__(self, file):
        self.walls, self.goals, self.start_positions = parse_maze_for_wavefront(file)
        self.maze =  Maze(file)
        self.paths = []
        self.file_name = file.split('/')[-1].split('.')[0]

    def calculate_shortest_paths(self):

        self.distances, self.previous_point, self.dimensions = wave_front_planner(self.walls, self.goals)
        start_pos = list(map(list, zip((self.start_positions['x'].to_numpy()).tolist(),
                                       (self.start_positions['y'].to_numpy()).tolist())))
        goal = list(map(list, zip((self.goals['x'].to_numpy()).tolist(),
                                       (self.goals['y'].to_numpy()).tolist())))[0]

        for start_pos in start_pos:
            self.paths.append(
                Path(
                    start_pos[0], start_pos[1], goal[0], goal[0],
                    get_path(start_pos[0], start_pos[1], self.distances, self.previous_point, self.dimensions)
                )
            )

    def save(self):
        with open('Paths/'+self.file_name + '_paths.pkl', 'wb') as out_file:
            pickle.dump(self, out_file)


class Path:
    def __init__(self, start_x, start_y, goal_x, goal_y, path):
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.path = path
