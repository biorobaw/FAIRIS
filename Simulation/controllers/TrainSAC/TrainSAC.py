import os

os.chdir("../../..")

from reinforcement_lib.SAC.Agent import Agent
import numpy as np

env = WebotsEnv(maze_file='WM00', pc_network_name='uniform_test', max_steps_per_episode=200)

agent = Agent(input_dims=env.observation_spec().shape, env=env, n_actions=env.action_space.n, batch_size=32, ver_name='V0')
n_episodes = 1000

best_score = 0
score_history = []
load_checkpoint = False

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
    if score >= best_score or avg_score >= best_score - 1.5:
        agent.save_models()
        best_score = score
    print(f'Episode: {episode}, Score: {score}, Avg Score: {avg_score}')
    agent.memory.increase_prop_sequence()


env.reset_environment()