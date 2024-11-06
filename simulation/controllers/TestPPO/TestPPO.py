import os

os.chdir("../../..")

import numpy as np
from reinforcement_lib.PPO import Agent
from fairis_tools.experiment_tools import ExperimentLogger, EpisodeLogger

maze_file = 'WM10'
pc_network_name = 'uniform_25'

env = WebotsEnv(maze_file=maze_file,
                pc_network_name=pc_network_name,
                max_steps_per_episode=300,
                action_length=0.3)
load_checkpoint = True
ver_name = pc_network_name +'_'+maze_file+'_rewardV3'
N = 20
batch_size = 10
n_epochs = 4
alpha = 0.0003
gamma = 0.95
experiment_logger = ExperimentLogger(maze_file,'uniform_25')

agent = Agent(n_actions=env.action_space.n,
                      batch_size=batch_size,
                      gamma=gamma,
                      alpha=alpha,
                      n_epochs=n_epochs,
                      input_dims=env.observation_spec().shape,
                      ver_name=ver_name)

if load_checkpoint:
    agent.load_models()
    env.set_mode(mode='train',
                 noise=False,
                 noise_intensity=0.01,
                 PC_decay=False,
                 PC_decay_percent=0.1)

n_episodes = 100

best_score = 0.0
score_history = []
avg_score = 0

for i in range(n_episodes):
    observation = env.reset()
    path_length = 0
    done = False
    score = 0
    start_x, start_y, start_theta = env.get_robot_pose()
    episode_logger = EpisodeLogger(start_x, start_y)
    prev_action = None

    while not done:
        action, prob, val = agent.choose_action(observation,prev_action,bias=.1)
        prev_action = action
        observation_, reward, done, info = env.step(action)
        episode_logger.log_step(robot_x=info[0],robot_y=info[1])
        path_length += 1
        score += reward
        # agent.remember(observation, action, prob, val, reward, done)
        observation = observation_

    experiment_logger.log_episode(episode_logger,score,path_length)
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'Path Length: ', path_length)

# with open('data/'+maze_file+'_test_10_decay_V3','wb') as file:
#     pickle.dump(experiment_logger,file)

env.reset_environment()