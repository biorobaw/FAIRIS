import math
from threading import Thread

import numpy as np
from sklearn.neighbors import KDTree


class PlaceCellTread(Thread):
    def __init__(self,pc,robot_x,robot_y):
        Thread.__init__(self)
        self.pc = pc
        self.robot_x = robot_x
        self.robot_y = robot_y
    def run(self):
        self.pc.calculate_activation(self.robot_x,self.robot_y)

class PlaceCellGenParm:
    def __init__(self,num_of_pc = 1, pc_scales = [.16]):
        self.num_of_pc = num_of_pc
        self.pc_scales = pc_scales

class PlaceCell:
    def __init__(self,id,x,y,r,alpha = 0.01,min_activation=0.001):
        self.id = id
        self.center_x = x
        self.center_y = y
        self.radius = r
        self.alpha = alpha
        self.min_activation = min_activation
        self.radius_tolerance = self.radius + self.alpha
        self.radius_squared = self.radius_tolerance**2
        self.k = math.log(self.min_activation)/self.radius_squared
        self.activity = 0.0


    def calculate_activation(self,robot_x,robot_y):
        """
        Computes the activation of the place cell.
        @param robot_x: The robot's current x coordinate
        @param robot_y: The robot's current y coordinate
        """
        dx = self.center_x - robot_x
        dy = self.center_y - robot_y
        r2 = dx**2 + dy**2
        self.activity = math.exp(self.k * r2)
        # if r2 <= self.radius_squared:
        #     self.activity = math.exp(self.k * r2)
        # else:
        #     self.activity = 0.0


class PlaceCellNetwork:

    def __init__(self,pc_generation_parm = PlaceCellGenParm()):
        self.pc_network = None
        self.pc_list = []
        self.pc_coordinates = []
        self.pc_generation_parm = pc_generation_parm
        self.add_pc_to_network(0.0,0.0,radius=4.243)

    def add_pc_to_network(self, robot_x, robot_y,radius=.1):
        pc_id = len(self.pc_list)
        self.pc_list.append(PlaceCell(pc_id, robot_x, robot_y, radius))
        self.pc_coordinates.append((robot_x, robot_y))
        self.pc_network = KDTree(self.pc_coordinates)

    def get_num_active_pc(self, robot_x, robot_y, search_radius=.5):
        if self.pc_network is not None:
            return self.pc_network.query_radius([(robot_x,robot_y)], r=search_radius,count_only=True)[0]
        else:
            return 0

    # Used for threading implementation
    def calculate_single_pc_activation(self,pc,robot_x,robot_y):
        pc.calculate_activation(robot_x,robot_y)

    def activate_pc_network(self, robot_x, robot_y):
        # Standard approach
        for pc in self.pc_list:
            pc.calculate_activation(robot_x,robot_y)

        # Threading approach
        # pc_activation_threads = []
        # for pc in self.pc_list:
        #     t = PlaceCellTread(pc,robot_x,robot_y)
        #     pc_activation_threads.append(t)
        #     t.start()
        # for t in pc_activation_threads:
        #     t.join()
        #     self.pc_list[t.pc.id] = t.pc

    def get_total_pc_activation(self):
        sum = 0
        for pc in self.pc_list:
            sum += pc.activity
        return sum

    def get_all_pc_activations_normalized(self, robot_x, robot_y):
        self.activate_pc_network(robot_x=robot_x,robot_y=robot_y)
        self.normilize_all_pc()
        return np.array([pc.activity for pc in self.pc_list],dtype=np.float32)

    def print_pc_activations(self):
        for pc in self.pc_list:
            print(pc.id, pc.activity)
    def normilize_all_pc(self):
        total_activation = self.get_total_pc_activation()
        if total_activation == 0:
            total_activation = 1
        for pc in self.pc_list:
            pc.activity = pc.activity/total_activation

class VisiualPlaceCell:
    def __init__(self, pc_id, center_point, sigma):
        self.id = pc_id
        self.center_point = center_point
        self.sigma = sigma
        self.activity = 0.0

    def calculate_activation(self,robot_point):
        distance = np.linalg.norm(self.center_point - robot_point)
        self.activity = np.exp(-(distance**2) / (2*self.sigma**2))

class VisualPlaceCellNetwork:
    def __init__(self):
        self.pc_network = None
        self.pc_list = []
        self.pc_coordinates = []

    def add_pc_to_network(self, robot_point, radius=.1):
        pc_id = len(self.pc_list)
        self.pc_list.append(VisiualPlaceCell(pc_id, robot_point, radius))
        self.pc_coordinates.append(robot_point)
        self.pc_network = KDTree(self.pc_coordinates)

    def get_num_active_pc(self, robot_point, search_radius=.5):
        if self.pc_network is not None:
            return self.pc_network.query_radius([robot_point], r=search_radius, count_only=True)[0]
        else:
            return 0

    def calculate_single_pc_activation(self, pc, robot_point):
        pc.calculate_activation(robot_point)

    def activate_pc_network(self, robot_point):
        # Standard approach
        for pc in self.pc_list:
            pc.calculate_activation(robot_point)
        self.normilize_all_pc()
    def get_total_pc_activation(self):
        total_activation = 0
        for pc in self.pc_list:
            total_activation += pc.activity
        return total_activation

    def get_all_pc_activations_normalized(self, robot_point,norm_type = 'norm'):
        self.activate_pc_network(robot_point)
        if norm_type == 'norm':
            self.normilize_all_pc()
        elif norm_type == 'min_max':
            self.min_max_normalize_pc_activations()
        return np.array([pc.activity for pc in self.pc_list], dtype=np.float32)

    def print_pc_activations(self):
        for pc in self.pc_list:
            print(pc.id, pc.activity)

    def normilize_all_pc(self):
        total_activation = self.get_total_pc_activation()
        if total_activation == 0:
            total_activation = 1
        for pc in self.pc_list:
            pc.activity = pc.activity / total_activation

    def min_max_normalize_pc_activations(self):
        """
        Perform min-max normalization on the place cell activations such that
        the minimum activation becomes 0 and the maximum becomes 1.
        """
        activations = np.array([pc.activity for pc in self.pc_list])
        min_activation = np.min(activations)
        max_activation = np.max(activations)

        if max_activation - min_activation == 0:
            # Prevent division by zero in case all activations are the same
            for pc in self.pc_list:
                pc.activity = 1.0  # Set all activations to 1 (arbitrary choice)
        else:
            for pc in self.pc_list:
                pc.activity = (pc.activity - min_activation) / (max_activation - min_activation)