import os

os.chdir("../../..")

import numpy as np
from reinforcement_lib.SAC.Agent import Agent

env = WebotsEnv(maze_file='WM00', pc_network_name='uniform_test', max_steps_per_episode=200)
env.set_mode(mode='test')

agent = Agent(input_dims=env.observation_spec().shape, env=env, n_actions=env.action_space.n, batch_size=32,ver_name='Vs')
n_episodes = 4

best_score = 0
score_history = []
load_checkpoint = True

if load_checkpoint:
    agent.load_models()

for episode in range(n_episodes):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        next_observation, reward, done, info = env.step(action)
        score += reward
        agent.remember(observation, action, reward, next_observation, done)
        if not load_checkpoint:
            agent.learn()
        observation = next_observation
    score_history.append(score)

    avg_score = np.mean(score_history[-100:])
    print(f'Episode: {episode}, Score: {score}, Avg Score: {avg_score}')

env.reset_environment()
