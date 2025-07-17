import operator
from reinforcement_lib.reinforcement_utils.model_functions import *
from fairis_tools.experiment_tools.image_processing.image_feature_lib import *
from fairis_lib.simulation_lib.environment import Maze
from controller import Supervisor
from matplotlib import patches
import math
from fairis_lib.robot_lib.robot_tools import *
from fairis_tools.experiment_tools.image_processing.feature_extractor import FeatureExtractor
class HamBot(Supervisor):

    # Initiilize an instance of Webots Harrison's RosBot
    def __init__(self, action_length=0.5,enable_cnn_features=False,cnn_extractor_model=None):

        # Inherent from Webots Robot Class: https://cyberbotics.com/doc/reference/robot
        self.experiment_supervisor = Supervisor()

        # Add a display to plot the place cells as they are generated
        self.pc_display = self.experiment_supervisor.getDevice('Place Cell Display')
        self.pc_display.setOpacity(1.0)

        # Sets Supervisor Root Nodes
        self.root_node = self.experiment_supervisor.getRoot()
        self.children_field = self.root_node.getField('children')
        self.robot_node = self.experiment_supervisor.getSelf()
        self.robot_translation_field = self.robot_node.getField('translation')
        self.robot_rotation_field = self.robot_node.getField('rotation')

        # Physical Robot Specifications
        self.wheel_radius = .045  # meters
        self.axel_length = .184  # meters
        self.robot_radius = .3086  # m
        self.action_length = action_length
        self.action_set = {
            0: [0, self.action_length],
            1: [45, self.action_length],
            2: [90, self.action_length],
            3: [135, self.action_length],
            4: [180, self.action_length],
            5: [225, self.action_length],
            6: [270, self.action_length],
            7: [315, self.action_length],
        }

        # Experiment Variables
        self.previous_action_index = -1

        # Define all systems and makes them class attributes
        self.timestep = int(self.experiment_supervisor.getBasicTimeStep())

        # Webots Rotational Motors: https://cyberbotics.com/doc/reference/motor
        self.left_motor = self.experiment_supervisor.getDevice('left motor')
        self.right_motor = self.experiment_supervisor.getDevice('right motor')
        self.all_motors = [self.left_motor, self.right_motor]
        self.max_motor_velocity = self.left_motor.getMaxVelocity()

        # Initialize robot's motors
        for motor in self.all_motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(0.0)

        # Webots Wheel Positional Sensors: https://www.cyberbotics.com/doc/reference/positionsensor
        self.left_encoder = self.experiment_supervisor.getDevice('left wheel encoder')
        self.right_encoder = self.experiment_supervisor.getDevice('right wheel encoder')
        self.all_encoders = [self.left_encoder, self.right_encoder]

        # Initialize robot's encoders
        for encoder in self.all_encoders:
            encoder.enable(self.timestep)


        # Webots Camera: https://cyberbotics.com/doc/reference/camera
        self.camera = self.experiment_supervisor.getDevice('camera')
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)

        # Webots RpLidarA2: https://www.cyberbotics.com/doc/guide/lidar-sensors#slamtec-rplidar-a2
        self.lidar = self.experiment_supervisor.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        # Webots IMU: https://www.cyberbotics.com/doc/guide/imu-sensors#mpu-9250
        # Webots IMU Accelerometer: https://www.cyberbotics.com/doc/reference/accelerometer
        self.imu = self.experiment_supervisor.getDevice('imu')
        self.imu.enable(self.timestep)

        # Webots GPS:
        self.gps = self.experiment_supervisor.getDevice('gps')
        self.gps.enable(self.timestep)

        self.sensor_calibration()

        # Enable CNN-based feature extraction if required
        self.enable_cnn_features = enable_cnn_features
        self.cnn_extractor_model = cnn_extractor_model
        if self.enable_cnn_features:
            self.cnn_feature_extractor = FeatureExtractor(self.cnn_extractor_model)
        else:
            self.cnn_feature_extractor = None

    # Preforms one timestep to update all sensors should be used when initializing robot and after teleport
    def sensor_calibration(self):
        while self.experiment_supervisor.step(self.timestep) != -1:
            break

    def get_robot_pose(self):
        while self.experiment_supervisor.step(self.timestep) != -1:
            current_x, current_y, current_z = self.robot_translation_field.getSFVec3f()
            break
        return current_x, current_y, self.get_bearing()

    def get_robot_pov_features(self, landmark_dictionary):
        """
        Get combined feature vector including CNN features, multimodal features, and robot pose.

        Args:
            landmark_dictionary (dict): Dictionary to identify landmarks in the POV image.

        Returns:
            np.ndarray: Combined feature vector suitable for clustering.
        """
        pov, landmark_mask = self.get_pov_image(landmark_dictionary)
        x, y, theta = self.get_robot_pose()

        # Extract CNN features if enabled
        if self.enable_cnn_features:
            cnn_features = self.cnn_feature_extractor.get_cnn_features(pov)
        else:
            cnn_features = np.array([])  # Empty array if CNN features are not enabled

        multimodal_features = extract_combined_features(pov, landmark_mask, theta)

        return multimodal_features, cnn_features, landmark_mask

    def get_pov_image(self, landmark_dictionary):
        while self.experiment_supervisor.step(self.timestep) != -1:
            landmark_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            pov = self.camera.getImageArray()
            landmarks = self.camera.getRecognitionObjects()
            for landmark in landmarks:
                color = landmark.getColors()
                color = [color[0], color[1], color[2]]
                key_list = list(landmark_dictionary.keys())
                val_list = list(landmark_dictionary.values())
                position = val_list.index(color)
                mask_index = key_list[position]
                landmark_mask[mask_index] = 1
            break
        return pov, landmark_mask

    # Reads the robot's IMU compass and return bearing in degrees
    #   North   -> 90
    #   East    -> 0 or 360
    #   South   -> 270
    #   West    -> 180
    def get_bearing(self):
        compass_reading = self.imu.getRollPitchYaw()
        bearing = math.degrees(compass_reading[-1])
        if bearing < 0.0:
            bearing += 360.0
        return round(bearing)

    def get_closest_action_index(self):
        return int((self.get_bearing() // 45))

    def get_min_lidar_reading(self):
        self.sensor_calibration()
        return min(self.lidar.getRangeImage())

    # Reads current encoder readings and return an array of encoder positions:
    #   [left, right]
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
        self.sensor_calibration()
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
            self.left_motor.setVelocity(1 * velocity)
            self.right_motor.setVelocity(-1 * velocity)

        elif 0 < delta <= 180 or -360 <= delta < -180:
            self.left_motor.setVelocity(-1 * velocity)
            self.right_motor.setVelocity(1 * velocity)

    # Sets motor speeds using PID to move the robot forward a desired distance in mm
    def forward_motion_with_encoder_PID(self, travel_distance, starting_encoder_position, K_p=100):
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

    # Sets Front Left Motor Velocity (rad/sec)
    def set_left_motor_velocity(self, velocity, suppress=False):
        self.left_motor.setVelocity(self.velocity_saturation(velocity, suppress=suppress))

    # Sets Front Right Motor Velocity (rad/sec)
    def set_right_motor_velocity(self, velocity, suppress=False):
        self.right_motor.setVelocity(self.velocity_saturation(velocity, suppress=suppress))

    # Sets all motors speed to 0
    def stop(self):
        for motor in self.all_motors:
            motor.setVelocity(0)

    def slow_stop(self):
        while self.experiment_supervisor.step(self.timestep) != -1:
            current_velocity = self.left_motor.getVelocity()
            if current_velocity > .1:
                for motor in self.all_motors:
                    motor.setVelocity(current_velocity / 4)
            else:
                self.stop()
                break

    # Sets all motors speed to velocity
    def go_forward(self, velocity=1):
        for motor in self.all_motors:
            motor.setVelocity(self.velocity_saturation(velocity))

    # Gets Current Front Left Motor Encoder Reading (meters)
    def get_left_motor_encoder_reading(self):
        return self.left_encoder.getValue()

    # Gets Current Front Right Motor Encoder Reading (meters)
    def get_right_motor_encoder_reading(self):
        return self.right_encoder.getValue()

    # Get Lidar Range Image (meters)
    def get_lidar_range_image(self):
        return self.lidar.getRangeImage()

    # Rotates the robot in place to face end_bearing and stops within margin_error (DEFAULT: +-.001)
    def rotate_to(self, end_bearing, margin_error=.0001):
        counter = 0
        while self.experiment_supervisor.step(self.timestep) != -1:
            self.rotation_PID(end_bearing)
            counter += 1
            if end_bearing - margin_error <= self.get_bearing() <= end_bearing + margin_error:
                self.stop()
                break

    # Rotates the robot by the amount degree. Only rotates until robot reaches the calculated end_bearing
    def rotate(self, degree, margin_error=.0001):
        start_bearing = self.get_bearing()
        end_bearing = start_bearing - degree
        if end_bearing > 360:
            end_bearing -= 360
        elif end_bearing < 0:
            end_bearing += 360
        self.rotate_to(end_bearing, margin_error=margin_error)

    # Moves the robot forward in a straight line by the amount distance (in mm)
    def move_forward_with_PID(self, distance, margin_error=.1):
        forward_lidar_window = 15
        starting_encoder_position = self.get_encoder_readings()
        while self.experiment_supervisor.step(self.timestep) != -1:
            self.forward_motion_with_encoder_PID(distance, starting_encoder_position)
            if (distance - margin_error <=
                    self.calculate_wheel_distance_traveled(starting_encoder_position) <= distance + margin_error):
                self.slow_stop()
                break
            if (min(self.lidar.getRangeImage()[180-forward_lidar_window:180+forward_lidar_window]) < .4):
                self.slow_stop()
                break

    # Moves the robot forward in a straight line by the amount distance (in mm)
    def move_forward_no_PID(self, distance, velocity=20, margin_error=.01):
        forward_lidar_window = 15
        starting_encoder_position = self.get_encoder_readings()
        while self.experiment_supervisor.step(self.timestep) != -1:
            self.go_forward(velocity=velocity)
            if (distance - margin_error <=
                    self.calculate_wheel_distance_traveled(starting_encoder_position) <= distance + margin_error):
                self.stop()
                break
            if (min(self.lidar.getRangeImage()[180-forward_lidar_window:180+forward_lidar_window]) < .25):
                self.stop()
                break

    # Moves the robot to the point (x,y) by rotating and then moving in a straight line
    def move_to_xy_with_PID(self, x, y, margin_error=.1):
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
        print(motion_vector)
        if not (motion_vector[0] - margin_error <=
                self.get_bearing() <=
                motion_vector[0] + margin_error):
            self.rotate_to(motion_vector[0])

        self.move_forward_no_PID(motion_vector[1])

    def perform_random_action(self, bias=True):

        available_actions = [int(i) for i in self.get_possible_actions()]
        # Add motion Bias and normalize
        if bias:
            action_distribution = apply_softmax(add_motion_bias(available_actions, self.previous_action_index))
        else:
            action_distribution = apply_softmax(available_actions)
        random_action_index = np.random.choice(8, 1, p=action_distribution)[0]

        if self.check_if_action_is_possible(random_action_index):
            random_action = self.action_set.get(random_action_index)
            self.rotate_to(random_action[0])
            self.move_forward_with_PID(random_action[1])
        else:
            random_action_index = np.argmax(action_distribution)
            random_action = self.action_set.get(random_action_index)
            self.rotate_to(random_action[0])
            self.move_forward_with_PID(random_action[1])

        self.previous_action_index = random_action_index
        return random_action_index

    def perform_habituation_action(self, bias=True):
        available_actions = [int(i) for i in self.get_possible_actions()]
        # Add motion Bias and normalize
        if bias:
            action_distribution = apply_softmax(add_motion_bias(available_actions, self.previous_action_index))
        else:
            action_distribution = apply_softmax(available_actions)
        random_action_index = np.random.choice(8, 1, p=action_distribution)[0]
        if self.check_if_action_is_possible(random_action_index):
            self.perform_training_action_teleport(random_action_index)
        else:
            random_action_index = np.argmax(action_distribution)
            self.perform_training_action_teleport(random_action_index)

        self.previous_action_index = random_action_index
        self.sensor_calibration()
        return random_action_index

    def perform_action_with_PID(self, action_index):
        action = self.action_set.get(action_index)
        if self.check_if_action_is_possible(action_index=action_index):
            self.rotate_to(action[0])
            self.move_forward_with_PID(action[1])
        else:
            print("cant preform action")

    def perform_action_no_PID(self, action_index):
        action = self.action_set.get(action_index)
        if self.check_if_action_is_possible(action_index=action_index):
            self.rotate_to(action[0])
            self.move_forward_no_PID(500 * action[1])
        else:
            print("cant preform action")

    def perform_training_action(self, action_index):
        action = self.action_set.get(action_index)
        self.rotate_to(action[0])
        self.move_forward_with_PID(action[1])

    def perform_training_action_teleport(self, action_index):
        action = self.action_set.get(action_index)
        curr_x, curr_y, curr_theta = self.get_robot_pose()
        action_theta = math.radians(action[0])
        new_x = curr_x + self.action_length * math.cos(action_theta)
        new_y = curr_y + self.action_length * math.sin(action_theta)
        self.teleport_robot(x=new_x, y=new_y, theta=action_theta)
        return self.get_robot_pose()

    def get_possible_actions(self):
        min_action_distance = self.action_length + 0.25
        while self.experiment_supervisor.step(self.timestep) != -1:
            relative_distances = RelativeDistances(lidar_range_image=self.lidar.getRangeImage())
            available_actions = [0] * 8
            bin_index = 0
            front_action_index = self.get_closest_action_index()
            for bin in relative_distances.distance_bins:
                action_index = (front_action_index - bin_index) % 8
                available_actions[action_index] = min(bin) > min_action_distance
                bin_index += 1

            return available_actions

    def get_possible_training_actions(self):
        available_actions = self.get_possible_actions()
        return [i for i in range(len(available_actions)) if available_actions[i]]

    def get_possible_training_action_mask(self):
        available_actions = np.array(self.get_possible_actions())
        return np.array(np.multiply(available_actions, 1), dtype=np.float32)

    def check_if_action_is_possible(self, action_index=-1):
        forward_lidar_window = 30
        min_action_distance = .5
        if action_index == -1:
            if min(self.lidar.getRangeImage()[180-forward_lidar_window:180+forward_lidar_window]) > min_action_distance:
                return True
            else:
                return False
        else:
            action = self.action_set.get(action_index)
            self.rotate_to(action[0])
            if min(self.lidar.getRangeImage()[180-forward_lidar_window:180+forward_lidar_window]) > min_action_distance:
                return True
            else:
                return False

    # Supervisor Functions: allows robot to control the simulation
    # DO NOT MODIFY: unless you are attempting to manipulate the webots world simulations!!!

    # Takes in a xml maze file and creates the walls, starting locations, and goal locations
    def load_environment(self, maze_file):
        self.maze = Maze(maze_file, display_width=self.pc_display.getWidth(),
                         display_height=self.pc_display.getHeight())
        self.pc_figure, self.pc_figure_ax = self.maze.get_maze_figure()
        self.pc_figure.savefig('data/DataCache/temp.png')

        while self.experiment_supervisor.step(self.timestep) != -1:
            ir = self.pc_display.imageLoad('data/DataCache/temp.png')
            self.pc_display.imagePaste(ir, 0, 0, True)
            break

        self.obstical_nodes = []
        self.boundry_wall_nodes = []
        self.cylinder_landmark_nodes = []
        self.tag_landmark_nodes = []

        for obstacles in self.maze.obstacles:
            self.children_field.importMFNodeFromString(-1, obstacles.get_webots_node_string())
            self.obstical_nodes.append(self.experiment_supervisor.getFromDef('Obstacle'))
        for boundary_wall in self.maze.boundary_walls:
            self.children_field.importMFNodeFromString(-1, boundary_wall.get_webots_node_string())
            self.boundry_wall_nodes.append(self.experiment_supervisor.getFromDef('Obstacle'))
        for cylinder_landmark in self.maze.cylinder_landmarks:
            self.children_field.importMFNodeFromString(-1, cylinder_landmark.get_webots_node_string())
            self.cylinder_landmark_nodes.append(self.experiment_supervisor.getFromDef('Landmark'))
        for tag_landmark in self.maze.tag_landmarks:
            self.children_field.importMFNodeFromString(-1, tag_landmark.get_webots_node_string())
            self.tag_landmark_nodes.append(self.experiment_supervisor.getFromDef('RectangularPanel'))

    def reset_environment(self):
        self.teleport_robot(theta=math.pi / 2)
        total_nodes = len(self.obstical_nodes) + len(self.boundry_wall_nodes) + len(self.landmark_nodes)
        for i in range(total_nodes):
            self.children_field.removeMF(-1)
        self.maze.close_maze_figure()
        self.sensor_calibration()

    # Teleports the robot to the point (x,y,z)
    def teleport_robot(self, x=0.0, y=0.0, z=0.034, theta=math.pi):
        self.robot_translation_field.setSFVec3f([x, y, z])
        self.robot_rotation_field.setSFRotation([0, 0, 1, theta])
        self.sensor_calibration()

    def move_to_training_start(self):
        starting_position = self.maze.experiment_starting_location[0]
        self.teleport_robot(starting_position.x, starting_position.y, theta=starting_position.theta)

    # Moves the robot to a random starting position
    def move_to_testing_start(self, index=-1):
        if index == -1:
            starting_position = self.maze.get_random_experiment_testing_starting_position()
        else:
            starting_position = self.maze.experiment_starting_location[index]
        self.teleport_robot(starting_position.x, starting_position.y, theta=starting_position.theta)
        return self.get_robot_pose()

    # Moves the robot to a random starting position
    def move_to_random_experiment_start(self):
        starting_position = self.maze.get_random_experiment_starting_position()
        self.teleport_robot(starting_position.x, starting_position.y, theta=starting_position.theta)
        return self.get_robot_pose()

    # Moves the robot to a random starting position
    def move_to_habituation_start(self, index=-1):
        if index == -1:
            starting_position = self.maze.get_random_habituation_starting_position()
        else:
            starting_position = self.maze.habituation_start_location[index]
        self.teleport_robot(starting_position.x, starting_position.y, theta=starting_position.theta)
        return self.get_robot_pose()

    def check_at_goal(self):
        current_x, current_y, currentcurrent_z = self.robot_translation_field.getSFVec3f()
        goal_x, goal_y = self.maze.get_goal_location()
        distance_to_goal = math.sqrt((current_x - goal_x) ** 2 + (current_y - goal_y) ** 2)
        return distance_to_goal < 0.8

    def get_dist_to_goal(self):
        current_x, current_y, currentcurrent_z = self.robot_translation_field.getSFVec3f()
        goal_x, goal_y = self.maze.get_goal_location()
        distance_to_goal = math.sqrt((current_x - goal_x) ** 2 + (current_y - goal_y) ** 2)
        return distance_to_goal

    def show_loaded_pc_network(self, pc_network):
        for pc in pc_network.pc_list:
            self.update_pc_display(pc)

    # Plots Place cells and shows them on the Display
    def update_pc_display(self, place_cell):
        new_pc = patches.Circle((place_cell.center_x, place_cell.center_y), radius=place_cell.radius, fill=False)
        self.pc_figure_ax.add_patch(new_pc)
        self.pc_figure_ax.set_ylim(-4.25, 4.25)
        self.pc_figure_ax.set_xlim(-4.25, 4.25)
        self.pc_figure.savefig('data/DataCache/temp.png')
        while self.experiment_supervisor.step(self.timestep) != -1:
            ir = self.pc_display.imageLoad('data/DataCache/temp.png')
            self.pc_display.imagePaste(ir, 0, 0, True)
            break
