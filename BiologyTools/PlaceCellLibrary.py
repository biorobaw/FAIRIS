import math
from threading import Thread

from sklearn.neighbors import KDTree


class PlaceCellTread(Thread):
    def __init__(self, pc, robot_x, robot_y):
        Thread.__init__(self)
        self.pc = pc
        self.robot_x = robot_x
        self.robot_y = robot_y

    def run(self):
        self.pc.calculate_activation(self.robot_x, self.robot_y)


class PlaceCellGenParm:
    def __init__(self, num_of_pc=1, pc_scales=None):
        if pc_scales is None:
            pc_scales = [.16]

        self.num_of_pc = num_of_pc
        self.pc_scales = pc_scales


class PlaceCell:
    def __init__(self, id, x, y, r, alpha=0.01, min_activation=0.001):
        self.id = id
        self.center_x = x
        self.center_y = y
        self.radius = r
        self.alpha = alpha
        self.min_activation = min_activation
        self.radius_tolerance = self.radius + self.alpha
        self.radius_squared = self.radius_tolerance ** 2
        self.k = math.log(self.min_activation / self.radius_squared)
        self.activity = 0.0

    def calculate_activation(self, robot_x, robot_y):
        """
        Computes the activation of the place cell.
        @param robot_x: The robot's current x coordinate
        @param robot_y: The robot's current y coordinate
        """
        dx = self.center_x - robot_x
        dy = self.center_y - robot_y
        r2 = dx ** 2 + dy ** 2
        if r2 <= self.radius_squared:
            self.activity = math.exp(self.k * r2)
        else:
            self.activity = 0.0


class PlaceCellNetwork:

    def __init__(self, pc_generation_parm=PlaceCellGenParm()):
        self.pc_network = None
        self.pc_list = []
        self.pc_coordinates = []
        self.pc_generation_parm = pc_generation_parm
        self.add_pc_to_network(0.0,0.0,radius=4.243)

    def add_pc_to_network(self, robot_x, robot_y, radius=.1):
        pc_id = len(self.pc_list)
        self.pc_list.append(PlaceCell(pc_id, robot_x, robot_y, radius))
        self.pc_coordinates.append((robot_x, robot_y))
        self.pc_network = KDTree(self.pc_coordinates)

    def get_num_active_pc(self, robot_x, robot_y, search_radius=.5):
        if self.pc_network != None:
            return self.pc_network.query_radius([(robot_x, robot_y)], r=search_radius, count_only=True)[0]
        else:
            return 0

    # Used for threading implementation
    def calculate_single_pc_activation(self, pc, robot_x, robot_y):
        print('here')
        pc.calculate_activation(robot_x, robot_y)

    def calculate_total_pc_activation(self, robot_x, robot_y):
        # Threading approch
        pc_activation_threads = []
        for pc in self.pc_list:
            t = PlaceCellTread(pc, robot_x, robot_y)
            pc_activation_threads.append(t)
            t.start()
        for t in pc_activation_threads:
            t.join()
            self.pc_list[t.pc.id] = t.pc

    def print_pc_activations(self):
        for pc in self.pc_list:
            print(pc.id, pc.activity)

    def normilize_all_pc(self):
        pass
