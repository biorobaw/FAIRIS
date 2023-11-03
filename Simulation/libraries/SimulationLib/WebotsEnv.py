import numpy as np
import torch
# torch.manual_seed(69)
# np.random.seed(420)
import pickle
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from Simulation.libraries.RobotLib.RosBot import RosBot
import gym
from gym.spaces import Discrete

maze_file_dir = 'Simulation/worlds/mazes/Experiment1/'

class WebotsEnv(py_environment.PyEnvironment):
    def __init__(self, maze_file, pc_network_name, max_steps_per_episode=200):
        self.maze_file = maze_file
        self.pc_network_name = pc_network_name
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.trial_counter = 0
        self.starting_permutaions = np.random.permutation(4)
        self.robot = RosBot()
        self.robot.load_environment(maze_file_dir + maze_file + '.xml')
        self.set_mode()
        self.action_space = Discrete(n=8)

        with open("Simulation/GeneratedPCNetworks/" + pc_network_name, 'rb') as pc_file:
            self.experiment_pc_network = pickle.load(pc_file)

        self.num_place_cells = len(self.experiment_pc_network.pc_list)

        # TF-Agents specs
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.num_place_cells + self.action_space.n,), dtype=np.float32, minimum=0,
                                                       maximum=1, name='sensor_data')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return array_spec.BoundedArraySpec(shape=(self.num_place_cells + self.action_space.n,), dtype=np.float32, minimum=0,
                                                       maximum=1, name='sensor_data')

    def _reset(self):
        self.current_step = 0
        if self.trial_counter % 4 == 0:
            self.starting_permutaions = np.random.permutation(4)
        starting_index = self.starting_permutaions[self.trial_counter % 4]
        if self.mode == 'train':
            robot_x, robot_y, robot_theta = self.robot.move_to_testing_start(index=starting_index)
        else:
            robot_x, robot_y, robot_theta = self.robot.move_to_habituation_start(index=starting_index)
        self.trial_counter += 1
        self.robot.experiment_supervisor.simulationResetPhysics()

        PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(robot_x, robot_y)
        avalible_actions = self.robot.get_possible_actions()
        initial_observation = np.concatenate([PC_Activations, avalible_actions]).astype(np.float32)
        # initial_observation = np.array(self.experiment_pc_network.get_all_pc_activations_normalized(robot_x, robot_y), dtype=np.float32)
        return initial_observation

    def _step(self, action):
        self.current_step += 1
        reward = 0
        if self.robot.check_if_action_is_possible(action):
            robot_x, robot_y, robot_theta = self.robot.perform_training_action_telaport(action)
        else:
            robot_x, robot_y, robot_theta = self.robot.get_robot_pose()
            reward += -2
        PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(robot_x, robot_y)
        avalible_actions = self.robot.get_possible_actions()

        # observation = np.array(self.experiment_pc_network.get_all_pc_activations_normalized(robot_x, robot_y), dtype=np.float32)

        observation = np.concatenate([PC_Activations, avalible_actions]).astype(np.float32)

        if self.robot.check_at_goal():
            reward += 10.0  # Explicitly making sure it's float.
            done = True
        elif self.current_step >= self.max_steps_per_episode:
            reward += 0.0  # Explicitly making sure it's float.
            done = True
        else:
            reward += -self.robot.get_dist_to_goal()/4.243
            done = False

        reward = np.array(reward, dtype=np.float32)


        return observation, reward, done, []

    def render(self, mode='human'):
        pass

    def close(self):
        self.robot.experiment_supervisor.simulationReset()

    def set_mode(self, mode = 'train'):
        self.mode = mode


# Gym registration
gym.register(
    id='Webots-v0',
    entry_point='WebotsEnv:WebotsEnv',
    kwargs={'maze_file': 'WM00',
            'pc_network_name': 'uniform_test',
            'max_steps_per_episode': 200}
)
