import operator

import matplotlib.pyplot as plt
import numpy as np

from Simulation.libraries.Environment import *
from controller import Supervisor

action_set = {
    0: [0, .8],
    1: [45, .8],
    2: [90, .8],
    3: [135, .8],
    4: [180, .8],
    5: [225, .8],
    6: [270, .8],
    7: [315, .8],
}


class RelativeDistances:
    def __init__(self, lidar_range_image):
        temp = [lidar_range_image[-60:], lidar_range_image[0:60]]
        self.rear_distances = []
        for r in temp:
            self.rear_distances += r
        self.front_right_distances = lidar_range_image[440:560]
        self.right_distances = lidar_range_image[540:660]
        self.rear_right_distances = lidar_range_image[640:760]
        self.front_distances = lidar_range_image[340:460]
        self.rear_left_distances = lidar_range_image[40:160]
        self.left_distances = lidar_range_image[140:260]
        self.front_left_distances = lidar_range_image[240:360]
        self.distance_bins = [self.front_distances,
                              self.front_right_distances,
                              self.right_distances,
                              self.rear_right_distances,
                              self.rear_distances,
                              self.rear_left_distances,
                              self.left_distances,
                              self.front_left_distances]


# Function to calculate the angle and distance between two points (x1,y1) and (x2,y2)
def calculate_motion_vector(x1, y1, x2, y2):
    theta = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
    if theta < 0:
        theta += 360
    magnitude = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 1000
    return np.array([theta, magnitude])


# Custom Class of RosBot offered by Webots
#   Webots: https://www.cyberbotics.com/doc/guide/rosbot
#   Sepecs: https://husarion.com/manuals/rosbot/#specification
class RosBot(Supervisor):

    # Initiilize an instance of Webots Harrison's RosBot
    def __init__(self):

        # Inherent from Webots Robot Class: https://cyberbotics.com/doc/reference/robot
        self.experiment_supervisor = Supervisor()

        # Add a display to plot the place cells as they are generated
        self.pc_display = self.experiment_supervisor.getDevice('Place Cell Display')
        self.pc_display.setOpacity(1.0)

        # Sets Supervisor Root Nodes
        self.root_node = self.experiment_supervisor.getRoot()
        self.children_field = self.root_node.getField('children')
        self.robot_node = self.experiment_supervisor.getFromDef('Agent')
        self.robot_translation_field = self.robot_node.getField('translation')

        # Physical Robot Specifications
        self.wheel_radius = 42.5  # mm
        self.axel_length = 265  # mm
        self.robot_radius = 308.6 / 2  # mm

        # Define all systems and makes them class atributes
        self.timestep = int(self.experiment_supervisor.getBasicTimeStep())

        # Webots Rotational Motors: https://cyberbotics.com/doc/reference/motor
        self.front_left_motor = self.experiment_supervisor.getDevice('front left wheel motor')
        self.front_right_motor = self.experiment_supervisor.getDevice('front right wheel motor')
        self.rear_left_motor = self.experiment_supervisor.getDevice('rear left wheel motor')
        self.rear_right_motor = self.experiment_supervisor.getDevice('rear right wheel motor')
        self.all_motors = [self.front_left_motor,
                           self.front_right_motor,
                           self.rear_left_motor,
                           self.rear_right_motor]
        self.max_motor_velocity = self.rear_left_motor.getMaxVelocity()

        # Initialize robot's motors
        for motor in self.all_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        # Webots Wheel Positional Sensors: https://www.cyberbotics.com/doc/reference/positionsensor
        self.front_left_encoder = self.experiment_supervisor.getDevice('front left wheel motor sensor')
        self.front_right_encoder = self.experiment_supervisor.getDevice('front right wheel motor sensor')
        self.rear_left_encoder = self.experiment_supervisor.getDevice('rear left wheel motor sensor')
        self.rear_right_encoder = self.experiment_supervisor.getDevice('rear right wheel motor sensor')
        self.all_encoders = [self.front_left_encoder,
                             self.front_right_encoder,
                             self.rear_left_encoder,
                             self.rear_right_encoder]

        # Initialize robot's encoders
        for encoder in self.all_encoders:
            encoder.enable(self.timestep)

        # Webots Astra Camera: https://cyberbotics.com/doc/guide/range-finder-sensors#orbbec-astra
        self.depth_camera = self.experiment_supervisor.getDevice('camera depth')
        self.depth_camera.enable(self.timestep)

        self.rgb_camera = self.experiment_supervisor.getDevice('camera rgb')
        self.rgb_camera.enable(self.timestep)

        # Webots RpLidarA2: https://www.cyberbotics.com/doc/guide/lidar-sensors#slamtec-rplidar-a2
        self.lidar = self.experiment_supervisor.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Webots IMU: https://www.cyberbotics.com/doc/guide/imu-sensors#mpu-9250
        # Webots IMU Accelerometer: https://www.cyberbotics.com/doc/reference/accelerometer
        self.accelerometer = self.experiment_supervisor.getDevice('imu accelerometer')
        self.accelerometer.enable(self.timestep)
        # Webots IMU Gyro: https://www.cyberbotics.com/doc/reference/gyro
        self.gyro = self.experiment_supervisor.getDevice('imu gyro')
        self.gyro.enable(self.timestep)
        # Webots IMU Compass: https://www.cyberbotics.com/doc/reference/compass
        self.compass = self.experiment_supervisor.getDevice('imu compass')
        self.compass.enable(self.timestep)

        # Webots GPS:
        self.gps = self.experiment_supervisor.getDevice('gps')
        self.gps.enable(self.timestep)

        # Webots Disance Sensors: https://www.cyberbotics.com/doc/reference/distancesensor
        self.front_left_ds = self.experiment_supervisor.getDevice('front left distance sensor')
        self.front_right_ds = self.experiment_supervisor.getDevice('front right distance sensor')
        self.rear_left_ds = self.experiment_supervisor.getDevice('rear left distance sensor')
        self.rear_right_ds = self.experiment_supervisor.getDevice('rear right distance sensor')

        self.all_distance_sensors = [self.front_left_ds,
                                     self.front_right_ds,
                                     self.rear_right_ds,
                                     self.rear_left_ds]

        for ds in self.all_distance_sensors:
            ds.enable(self.timestep)

        self.sensor_calibration()

    # Preforms one timestep to update all sensors should be used when initializing robot and after teleport
    def sensor_calibration(self):
        while self.experiment_supervisor.step(self.timestep) != -1:
            break

    def get_robot_pose(self):
        current_pose = self.gps.getValues()
        return current_pose[0], current_pose[1], self.get_bearing()

    # Sets all motors speed to 0
    def stop(self):
        for motor in self.all_motors:
            motor.setVelocity(0)

    # Sets all motors speed to 0
    def go_forward(self, velocity=1):
        for motor in self.all_motors:
            motor.setVelocity(velocity)

    # Reads the robot's IMU compass and return bearing in degrees
    #   North   -> 90
    #   East    -> 0 or 360
    #   South   -> 270
    #   West    -> 180
    def get_bearing(self):
        compass_reading = self.compass.getValues()
        rad = math.atan2(compass_reading[0], compass_reading[1]) + math.pi / 2
        bearing = (rad - math.pi / 2) / math.pi * 180.0
        if bearing < 0.0:
            bearing += 360.0
        return round(bearing)

    def get_closest_action_index(self):
        return int((self.get_bearing() // 45))

    # Reads current encoder readings and return an array of encoder positions:
    #   [front_left, front_right, rear_right, rear_right]
    def get_encoder_readings(self):
        return [readings.getValue() for readings in self.all_encoders]

    # Calculates the average mm all the wheels have traveled from a relative starting encoder reading
    def calculate_wheel_distance_traveled(self, starting_encoder_position):
        current_encoder_readings = self.get_encoder_readings()
        differences = list(map(operator.sub, current_encoder_readings, starting_encoder_position))
        average_differences = sum(differences) / len(differences)
        average_distance = average_differences * self.wheel_radius
        return average_distance

    # Calculates the vector needed to move the robot to the point (x,y)
    def calculate_robot_motion_vector(self, x, y):
        current_position = self.gps.getValues()[0:2]
        return calculate_motion_vector(current_position[0], current_position[1], x, y)

    # Caps the motor velocities to ensure PID calculations do no exceed motor speeds
    def velocity_saturation(self, motor_velocity):
        if motor_velocity > self.max_motor_velocity:
            return self.max_motor_velocity
        elif motor_velocity < -1 * self.max_motor_velocity:
            return -1 * self.max_motor_velocity
        else:
            return motor_velocity

    # Sets motor speeds using PID to turn robot to desired bearing
    def rotation_PID(self, end_bearing, K_p=1):
        delta = end_bearing - self.get_bearing()
        velocity = self.velocity_saturation(K_p * abs(delta))
        if -180 <= delta <= 0 or 180 < delta <= 360:
            self.front_left_motor.setVelocity(1 * velocity)
            self.rear_left_motor.setVelocity(1 * velocity)
            self.front_right_motor.setVelocity(-1 * velocity)
            self.rear_right_motor.setVelocity(-1 * velocity)
        elif 0 < delta <= 180 or -360 <= delta < -180:
            self.front_left_motor.setVelocity(-1 * velocity)
            self.rear_left_motor.setVelocity(-1 * velocity)
            self.front_right_motor.setVelocity(1 * velocity)
            self.rear_right_motor.setVelocity(1 * velocity)

    # Sets motor speeds using PID to move the robot forward a desired distance in mm
    def forward_motion_with_encoder_PID(self, travel_distance, starting_encoder_position, K_p=.2):
        delta = travel_distance - self.calculate_wheel_distance_traveled(starting_encoder_position)
        velocity = self.velocity_saturation(K_p * delta)
        for motor in self.all_motors:
            motor.setVelocity(velocity)

    # Sets the motor speeds using PID to move the robot to the point (x,y)
    def forward_motion_with_xy_PID(self, x, y, K_p=10):
        current_position = self.gps.getValues()[0:2]
        delta_x = x - current_position[0]
        delta_y = y - current_position[1]
        delta = (delta_x + delta_y) / 2
        velocity = self.velocity_saturation(K_p * delta)
        for motor in self.all_motors:
            motor.setVelocity(velocity)

    # Rotates the robot in place to face end_bearing and stops within margin_error (DEFAULT: +-.001)
    def rotate_to(self, end_bearing, margin_error=.00001):
        while self.experiment_supervisor.step(self.timestep) != -1:
            self.rotation_PID(end_bearing)
            if end_bearing - margin_error <= self.get_bearing() <= end_bearing + margin_error:
                self.stop()
                break

    # Rotates the robot by the amount degree. Only rotates until robot reaches the calculated end_bearing
    def rotate(self, degree, margin_error=.01):
        start_bearing = self.get_bearing()
        end_bearing = start_bearing - degree
        if end_bearing > 360:
            end_bearing -= 360
        elif end_bearing < 0:
            end_bearing += 360
        self.rotate_to(end_bearing, margin_error=margin_error)

    # Moves the robot forward in a straight line by the amount distance (in mm)
    def move_forward_with_PID(self, distance, margin_error=.01):
        starting_encoder_position = self.get_encoder_readings()
        while self.experiment_supervisor.step(self.timestep) != -1:
            self.forward_motion_with_encoder_PID(distance, starting_encoder_position)
            if (distance - margin_error <=
                self.calculate_wheel_distance_traveled(starting_encoder_position) <= distance + margin_error) \
                    or (min(self.lidar.getRangeImage()[300:500]) < .2):
                self.stop()
                break

    # Moves the robot forward in a straight line by the amount distance (in mm)
    def move_forward_no_PID(self, distance, velocity=20, margin_error=.01):
        starting_encoder_position = self.get_encoder_readings()
        while self.experiment_supervisor.step(self.timestep) != -1:
            self.go_forward(velocity=velocity)
            if (distance - margin_error <=
                self.calculate_wheel_distance_traveled(starting_encoder_position) <= distance + margin_error) \
                    or (min(self.lidar.getRangeImage()[300:500]) < .2):
                self.stop()
                break

    # Moves the robot to the point (x,y) by rotating and then moving in a straight line
    def move_to_xy_with_PID(self, x, y, margin_error=.01):
        motion_vector = self.calculate_robot_motion_vector(x, y)
        if not (motion_vector[0] - margin_error <=
                self.get_bearing() <=
                motion_vector[0] + margin_error):
            self.rotate_to(motion_vector[0])
        motion_vector = self.calculate_robot_motion_vector(x, y)
        self.move_forward_with_PID(motion_vector[1])

    # Moves the robot to the point (x,y) by rotating and then moving in a straight line
    def move_to_xy_no_PID(self, x, y, velocity=20, margin_error=.01):
        motion_vector = self.calculate_robot_motion_vector(x, y)
        if not (motion_vector[0] - margin_error <=
                self.get_bearing() <=
                motion_vector[0] + margin_error):
            self.rotate_to(motion_vector[0])
        motion_vector = self.calculate_robot_motion_vector(x, y)
        self.move_forward_no_PID(motion_vector[1])

    def perform_random_action(self):
        random_action_index = randint(0, 7)
        if self.check_if_action_is_possible(action_index=random_action_index):
            random_action = action_set.get(random_action_index)
            self.rotate_to(random_action[0])
            self.move_forward_with_PID(500 * random_action[1])
        else:
            print("cant preform action")

    def perform_action_with_PID(self, action_index):
        action = action_set.get(action_index)
        if self.check_if_action_is_possible(action_index=action_index):
            self.rotate_to(action[0])
            self.move_forward_with_PID(500 * action[1])
        else:
            print("cant preform action")

    def perform_action_no_PID(self, action_index):
        action = action_set.get(action_index)
        if self.check_if_action_is_possible(action_index=action_index):
            self.rotate_to(action[0])
            self.move_forward_no_PID(500 * action[1])
        else:
            print("cant preform action")

    def check_if_action_is_possible(self, action_index=-1):
        min_action_distance = .5
        while self.experiment_supervisor.step(self.timestep) != -1:
            print(min(self.lidar.getRangeImage()))
            if action_index == -1:
                if min(self.lidar.getRangeImage()[350:450]) > min_action_distance:
                    return True
                else:
                    return False
            else:
                relative_distances = RelativeDistances(lidar_range_image=self.lidar.getRangeImage())
                available_actions = []
                for bin in relative_distances.distance_bins:
                    available_actions.append(min(bin) > min_action_distance)
                bin_index = (self.get_closest_action_index() - action_index) % 8
                return available_actions[bin_index]

    # Supervisor Functions: allows robot to control the simulation

    # Takes in a xml maze file and creates the walls, starting locations, and goal locations
    def load_environment(self, maze_file):
        self.maze = Maze(maze_file)
        self.obstical_nodes = []
        self.boundry_wall_nodes = []
        for obsticals in self.maze.obstacle:
            self.children_field.importMFNodeFromString(-1, obsticals.get_webots_node_string())
            self.obstical_nodes.append(self.experiment_supervisor.getFromDef('Obstacle'))
        for boundry_wall in self.maze.boundary_walls:
            self.children_field.importMFNodeFromString(-1, boundry_wall.get_webots_node_string())
            self.boundry_wall_nodes.append(self.experiment_supervisor.getFromDef('Obstacle'))

    # Teleports the robot to the point (x,y,z)
    def teleport_robot(self, x=0.0, y=0.0, z=0.0):
        self.robot_translation_field.setSFVec3f([x, y, z])
        self.sensor_calibration()

    # Moves the robot to a random starting position
    def move_to_random_start(self):
        starting_position = self.maze.get_random_starting_position()
        self.teleport_robot(starting_position.x, starting_position.y)

    # Plots Place cells and shows them on the Display
    def update_pc_display(self):
        fig, ax = plt.subplots(figsize=(2, 4), facecolor='lightskyblue',
                               layout='constrained')
        fig.suptitle('Figure')
        ax.set_title('Axes', loc='left', fontstyle='oblique', fontsize='medium')
        fig.savefig('DataCache/temp.png')
        plt.close(fig)
        while self.experiment_supervisor.step(self.timestep) != -1:
            ir = self.pc_display.imageLoad('DataCache/temp.png')
            self.pc_display.imagePaste(ir, 0, 0, True)
            break
