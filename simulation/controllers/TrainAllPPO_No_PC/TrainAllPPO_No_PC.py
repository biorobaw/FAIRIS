import os

os.chdir("../../..")

import numpy as np
from reinforcement_lib.PPO.PPONetworks_no_PC import Agent

from fairis_tools.experiment_tools import calculate_dynamic_N
import pickle

maze_files = ['WM20']
memory = False

first_experiment = True
load_checkpoint = False

Data = []

for maze_file in maze_files:
    for version in range(10):

        pc_network_name = "no_pc_V"+str(version)
        if first_experiment:
            env = WebotsEnv(maze_file=maze_file,
                            max_steps_per_episode=300,
                            action_length=0.3)
            first_experiment = False
        else:
            env.reload(maze_file=maze_file)

        experiment_logger = ExperimentLogger(maze_file, 'No_PC')

        batch_size = 10
        n_epochs = 4
        alpha = 0.0003
        gamma = 0.95
        ver_name = pc_network_name +'_'+maze_file +'_reward_no_mem'

        agent = Agent(n_actions=env.action_space.n,
                      batch_size=batch_size,
                      gamma=gamma,
                      alpha=alpha,
                      n_epochs=n_epochs,
                      input_dims=env.observation_spec().shape,
                      ver_name=ver_name)

        if load_checkpoint:
            agent.load_models()
            env.set_mode(mode='test')

        n_episodes = 2000

        best_score = 0.0
        score_history = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0

        never_saved = True
        for current_episode in range(n_episodes):
            N =calculate_dynamic_N(current_episode,max_episode=1500, N_start=40, N_end=10)
            observation = env.reset()
            done = False
            prev_action = None
            score = 0
            path_length = 0
            start_x, start_y, start_theta = env.get_robot_pose()
            episode_logger = EpisodeLogger(start_x, start_y)
            while not done:
                action, prob, val = agent.choose_action(observation,prev_action)
                observation_, reward, done, info = env.step(action)
                episode_logger.log_step(robot_x=info[0],robot_y=info[1])
                prev_action = action
                n_steps += 1
                path_length += 1
                score += reward
                agent.remember(observation, action, prob, val, reward, done)
                if n_steps % N == 0 and not load_checkpoint:
                    agent.learn(memory=memory)
                    learn_iters += 1
                observation = observation_

            experiment_logger.log_episode(episode_logger,score,path_length)

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if (avg_score > best_score and not load_checkpoint) or (current_episode == n_episodes - 1 and (never_saved or avg_score > 93)):
                best_score = avg_score
                agent.save_models()
                never_saved = False

            print('episode', current_episode, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                    'time_steps', n_steps, 'learning_steps', learn_iters)

        Data.append(experiment_logger)
        env.reset_environment()
with open('data/LargeScale/AllPPOTrainData_WM20_no_pc','wb') as file:
    pickle.dump(Data,file)
env.close()
