import os

os.chdir("../../..")

from reinforcement_lib.SAC.Agent import Agent
import numpy as np

import pickle

maze_files = ['WM00', 'WM10', 'WM20']

pc_versions = ['0_0', '0_1', '0_2', '0_3', '0_4',
              '1_0', '1_1', '1_2', '1_3', '1_4',
              '2_0', '2_1', '2_2', '2_3', '2_4',]

first_experiment = True
load_checkpoint = False

Data = []

for maze_file in maze_files:
    for pc_version in pc_versions:
        pc_network_name = maze_file+'_'+pc_version
        if first_experiment:
            env = WebotsEnv(maze_file=maze_file, pc_network_name=pc_network_name, max_steps_per_episode=200)
            first_experiment = False
        else:
            env.reload(maze_file=maze_file,pc_network_name=pc_network_name)

        experiment_logger = ExperimentLogger(maze_file, pc_network_name)

        agent = Agent(input_dims=env.observation_spec().shape,
                      env=env,
                      n_actions=env.action_space.n,
                      batch_size=32,
                      ver_name=pc_network_name)

        if load_checkpoint:
            agent.load_models()
            env.set_mode(mode='test')

        n_episodes = 1000
        best_score = 0.0
        score_history = []

        for episode in range(n_episodes):
            observation = env.reset()
            done = False
            score = 0
            path_length=0
            start_x, start_y, start_theta = env.get_robot_pose()
            episode_logger = EpisodeLogger(start_x, start_y)

            while not done:
                action = agent.choose_action(observation)
                next_observation, reward, done, info = env.step(action)
                episode_logger.log_step(robot_x=info[0], robot_y=info[1])
                score += reward
                path_length += 1
                agent.remember(observation, action, reward, next_observation, done)
                if not load_checkpoint:
                    agent.learn()
                observation = next_observation

            experiment_logger.log_episode(episode_logger, score, path_length)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if score >= best_score or avg_score >= best_score - 1.5:
                agent.save_models()
                best_score = score

            print(f'Episode: {episode}, Score: {score}, Avg Score: {avg_score}')
            agent.memory.increase_prop_sequence()

        Data.append(experiment_logger)
        env.reset_environment()

with open('data/AllSACTrainData','wb') as file:
    pickle.dump(Data,file)

env.close()

