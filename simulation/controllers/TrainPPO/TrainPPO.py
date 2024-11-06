import os

os.chdir("../../..")

import numpy as np
from reinforcement_lib.PPO.PPONetworks import Agent
from fairis_lib.simulation_lib.webots_torch_environment import WebotsEnv

env = WebotsEnv(maze_file='WM10', pc_network_name='uniform_9', max_steps_per_episode=200)

load_checkpoint = False

N = 20
batch_size = 5
n_epochs = 4
alpha = 0.0003
ver_name = 'uniform_9_WM10'

print(env.observation_spec().shape)

agent = Agent(n_actions=env.action_space.n,
              batch_size=batch_size,
              alpha=alpha,
              n_epochs=n_epochs,
              input_dims=env.observation_spec().shape,
              ver_name=ver_name)

if load_checkpoint:
    agent.load_models()
    env.set_mode(mode='test')

n_episodes = 1000


best_score = 0.0
score_history = []

learn_iters = 0
avg_score = 0
n_steps = 0

for i in range(n_episodes):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        n_steps += 1
        score += reward
        agent.remember(observation, action, prob, val, reward, done)
        if n_steps % N == 0 and not load_checkpoint:
            agent.learn()
            learn_iters += 1
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score and not load_checkpoint:
        best_score = avg_score
        agent.save_models()

    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)

env.reset_environment()