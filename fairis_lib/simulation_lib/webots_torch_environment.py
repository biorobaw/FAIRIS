import numpy as np
from fairis_lib.simulation_lib.noise import apply_noise
import pickle
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from fairis_lib.robot_lib.hambot import HamBot
import gym
from gym.spaces import Discrete

landmark_dictionary = {
    0: [1.0, 0.0, 0.0],
    1: [0.0, 1.0, 0.0],
    2: [0.0, 0.0, 1.0],
    3: [1.0, 1.0, 0.0],
    4: [1.0, 0.0, 1.0],
    5: [0.0, 1.0, 1.0],
    6: [1.0, 0.5, 0.0],
    7: [0.5, 0.0, 0.5],
    8: [0.5, 0.5, 0.0],
}

class WebotsEnv(py_environment.PyEnvironment):
    def __init__(self, maze_file, pc_network_name, pc_type = "GPS",max_steps_per_episode=200, action_length = 0.5):
        self.maze_file = maze_file
        self.pc_network_name = pc_network_name
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.episode_counter = 0
        self.starting_permutaions = np.random.permutation(4)
        self.robot = HamBot(action_length=action_length)
        self.set_mode()
        self.action_space = Discrete(n=8)
        self.pc_type = pc_type

        if self.pc_type == 'GPS':
            self.maze_file_dir = 'simulation/worlds/mazes/Experiment1/'
            self.pc_dir = "data/GeneratedPCNetworks/GPSPlaceCells/"
            with open(self.pc_dir + pc_network_name, 'rb') as pc_file:
                self.experiment_pc_network = pickle.load(pc_file)

        else:
            self.maze_file_dir = 'simulation/worlds/mazes/VisualPlaceCells/'
            self.pc_dir = "data/GeneratedPCNetworks/VisualPlaceCells/"
            with open(self.pc_dir + pc_network_name, 'rb') as pc_file:
                self.experiment_pc_network = pickle.load(pc_file)

        self.num_place_cells = len(self.experiment_pc_network.pc_list)
        self.robot.load_environment(self.maze_file_dir + maze_file + '.xml')

        # TF-Agents specs
        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.float32,
                                                        minimum=0.0,
                                                        maximum=1.0,
                                                        name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.num_place_cells,),
                                                             dtype=np.float32,
                                                             minimum=0.0,
                                                             maximum=1.0,
                                                             name='sensor_data')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return array_spec.BoundedArraySpec(shape=(self.num_place_cells,),
                                           dtype=np.float32,
                                           minimum=0.0,
                                           maximum=1.0,
                                           name='sensor_data')

    def _reset(self):
        self.current_step = 0
        if self.episode_counter % 4 == 0:
            self.starting_permutaions = np.random.permutation(4)
        starting_index = self.starting_permutaions[self.episode_counter % 4]
        if self.mode == 'train':
            robot_x, robot_y, robot_theta = self.robot.move_to_testing_start(index=starting_index)
        else:
            robot_x, robot_y, robot_theta = self.robot.move_to_habituation_start(index=starting_index)
        self.episode_counter += 1
        self.robot.experiment_supervisor.simulationResetPhysics()

        if self.pc_type == "GPS":
            PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(robot_x, robot_y)
        else:
            robot_point = self.robot.get_robot_pov_processed(landmark_dictionary)
            PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(robot_point)
        avalible_actions = self.robot.get_possible_actions()
        # initial_observation = np.concatenate([PC_Activations, avalible_actions]).astype(np.float32)
        initial_observation = PC_Activations
        return initial_observation

    def _step(self, action):
        self.current_step += 1
        reward = 0
        if self.robot.check_if_action_is_possible(action):
            robot_x, robot_y, robot_theta = self.robot.perform_training_action_teleport(action)
        else:
            robot_x, robot_y, robot_theta = self.robot.get_robot_pose()
            reward += -2
        if self.noise_type is None:
            if self.pc_type == "GPS":
                PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(robot_x, robot_y)
            else:
                robot_point = self.robot.get_robot_pov_processed(landmark_dictionary)
                PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(robot_point)
        else:
            # TODO add noise for visual PC
            noisy_x, noisy_y = apply_noise(robot_x, robot_y, noise_type=self.noise_type, level=self.noise_level)
            PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(noisy_x, noisy_y)


        available_actions = self.robot.get_possible_actions()

        # observation = np.concatenate([PC_Activations, available_actions]).astype(np.float32)
        observation = PC_Activations

        if self.robot.check_at_goal():
            reward += 100.0  # Explicitly making sure it's float.
            done = True
        elif self.current_step >= self.max_steps_per_episode:
            reward += 0.0  # Explicitly making sure it's float.
            done = True
        else:
            reward += -1
            done = False

        reward = np.array(reward, dtype=np.float32)

        return observation, reward, done, [robot_x,robot_y]

    def get_robot_pose(self):
        return self.robot.get_robot_pose()
    def render(self, mode='human'):
        pass

    def reset_environment(self):
        self.robot.reset_environment()

    def close(self):
        self.robot.experiment_supervisor.simulationReset()
        self.robot.experiment_supervisor.simulationQuit(status=1)

    def reload(self, maze_file, pc_network_name):
        self.robot.load_environment(self.maze_file_dir + maze_file + '.xml')
        if self.pc_type == 'GPS':
            with open(self.pc_dir + pc_network_name, 'rb') as pc_file:
                self.experiment_pc_network = pickle.load(pc_file)

            self.num_place_cells = len(self.experiment_pc_network.pc_list)
        else:
            with open(self.pc_dir + pc_network_name, 'rb') as pc_file:
                self.experiment_pc_network = pickle.load(pc_file)

        self.num_place_cells = len(self.experiment_pc_network.pc_list)

        # TF-Agents specs
        self._action_spec = array_spec.BoundedArraySpec(shape=(),
                                                        dtype=np.float32,
                                                        minimum=0.0,
                                                        maximum=1.0,
                                                        name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.num_place_cells + self.action_space.n,),
                                                             dtype=np.float32,
                                                             minimum=0.0,
                                                             maximum=1.0,
                                                             name='sensor_data')

    def set_mode(self, mode='train',
                 noise_type=None,
                 noise_level=0,
                 ):
        self.mode = mode
        self.noise_type = noise_type
        self.noise_level = noise_level

# Gym registration
gym.register(
    id='Webots-v0',
    entry_point='WebotsEnv:WebotsEnv',
    kwargs={'maze_file': 'WM00',
            'pc_network_name': 'uniform_test',
            'max_steps_per_episode': 200}
)
