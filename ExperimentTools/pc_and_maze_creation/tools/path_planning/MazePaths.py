import pickle

from Simulation.libraries.SimulationLib.MazeAndPcsParcer import parse_maze_for_wavefront
from Simulation.libraries.SimulationLib.Environment import Maze
import numpy as np
from heapq import heappush, heappop
from shapely.geometry import Point, LineString


class MazePaths:
    def __init__(self, file):
        self.walls, self.goals, self.start_positions = parse_maze_for_wavefront(file)
        self.maze =  Maze(file)
        self.paths = []
        self.file_name = file.split('/')[-1].split('.')[0]

    def calculate_shortest_paths(self):
        # Define the 8 directional actions
        actions = self.define_actions()
        # Convert walls to a format that can be used for distance checking (e.g., LineString objects)
        wall_lines = self.convert_walls_to_lines(self.walls)
        for start_position in self.start_positions.itertuples():
            start_x, start_y = start_position.x, start_position.y
            goal_x, goal_y = self.goals.iloc[0]['x'], self.goals.iloc[0]['y']
            path = self.find_path(start_x, start_y, goal_x, goal_y, actions, wall_lines)
            if path:
                self.paths.append(Path(start_x, start_y, goal_x, goal_y, path))

    def define_actions(self):
        # Define 8 actions with 45-degree headings
        d_theta = np.pi / 4
        step_length = 0.3
        return [(step_length * np.cos(i * d_theta), step_length * np.sin(i * d_theta)) for i in range(8)]

    def convert_walls_to_lines(self, walls):
        # Convert wall data to LineString objects for distance calculations
        return [LineString([(row.x1, row.y1), (row.x2, row.y2)]) for _, row in walls.iterrows()]

    def heuristic(self, current, goal):
        # Euclidean distance
        return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)

    def is_valid_move(self, x, y, wall_lines):
        # Check if the move is at least 0.25 meters away from all walls
        point = Point(x, y)
        return all(point.distance(wall) >= 0.25 for wall in wall_lines)

    def find_path(self, start_x, start_y, goal_x, goal_y, actions, wall_lines):
        start = (start_x, start_y)
        goal = (goal_x, goal_y)

        # Priority queue for A*
        frontier = []
        heappush(frontier, (0, start))

        # Stores where we came from for each location
        came_from = {start: None}

        # Cost to reach each node
        cost_so_far = {start: 0}

        while frontier:
            current = heappop(frontier)[1]
            # Check if goal is reached
            if np.linalg.norm(np.array(current) - np.array(goal)) < 0.8:
                break

            for action in actions:
                next_pos = (current[0] + action[0], current[1] + action[1])
                new_cost = cost_so_far[current] + np.linalg.norm(action)

                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    if self.is_valid_move(next_pos[0], next_pos[1], wall_lines):
                        cost_so_far[next_pos] = new_cost
                        priority = new_cost + self.heuristic(next_pos, goal)
                        heappush(frontier, (priority, next_pos))
                        came_from[next_pos] = current

        # Reconstruct path
        # current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from.get(current)
            print(current)
            if current is None:
                # No path found
                return None
        path.append(start)  # optional: add start position
        path.reverse()  # optional: reverse the path to start-to-goal order
        return path

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
