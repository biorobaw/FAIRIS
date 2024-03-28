import numpy as np
import torch
from random import shuffle
import pickle
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from Simulation.libraries.RobotLib.RosBot import RosBot
import gym
from gym.spaces import Discrete

maze_file_dir = 'Simulation/worlds/mazes/Experiment1/'


class WebotsEnv(py_environment.PyEnvironment):
    def __init__(self, maze_file, pc_network_name, max_steps_per_episode=200, action_length = 0.5):
        self.maze_file = maze_file
        self.pc_network_name = pc_network_name
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0
        self.trial_counter = 0
        self.starting_permutaions = np.random.permutation(4)
        self.robot = RosBot(action_length=action_length)
        self.set_mode()
        self.action_space = Discrete(n=8)

        self.robot.load_environment(maze_file_dir + maze_file + '.xml')
        with open("Simulation/GeneratedPCNetworks/" + pc_network_name, 'rb') as pc_file:
            self.experiment_pc_network = pickle.load(pc_file)

        self.num_place_cells = len(self.experiment_pc_network.pc_list)

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
            reward += -5
        if not self.noise and not self.PC_decay:
            PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(robot_x, robot_y)
        elif self.noise:
            base_PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(robot_x, robot_y)
            noise = np.random.normal(0,self.noise_intensity,len(base_PC_Activations))
            PC_Activations = base_PC_Activations + noise
        elif self.PC_decay:
            base_PC_Activations = self.experiment_pc_network.get_all_pc_activations_normalized(robot_x, robot_y)
            PC_Activations = base_PC_Activations * self.dead_pc_mask

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
            # reward += -self.robot.get_dist_to_goal()
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
        self.robot.load_environment(maze_file_dir + maze_file + '.xml')
        with open("Simulation/GeneratedPCNetworks/" + pc_network_name, 'rb') as pc_file:
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
                 noise=False,
                 noise_intensity=.1,
                 PC_decay=False,
                 PC_decay_percent=.10):
        self.mode = mode
        self.noise = noise
        self.noise_intensity = noise_intensity
        self.PC_decay = PC_decay
        self.PC_decay_percent = PC_decay_percent
        if self.PC_decay:
            num_dead_pc = int(len(self.experiment_pc_network.pc_list) * self.PC_decay_percent)
            num_alive_pc = len(self.experiment_pc_network.pc_list) - num_dead_pc
            self.dead_pc_mask = np.concatenate((np.ones(num_alive_pc, dtype=np.float32) , np.zeros(num_dead_pc,dtype=np.float32)), axis = 0)
            shuffle(self.dead_pc_mask)



# Gym registration
gym.register(
    id='Webots-v0',
    entry_point='WebotsEnv:WebotsEnv',
    kwargs={'maze_file': 'WM00',
            'pc_network_name': 'uniform_test',
            'max_steps_per_episode': 200}
)
