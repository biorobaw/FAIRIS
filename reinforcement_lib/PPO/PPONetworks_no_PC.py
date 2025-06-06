import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# class PPOMemory:
#     def __init__(self, batch_size):
#         self.states = []
#         self.probs = []
#         self.vals = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#
#         self.batch_size = batch_size
#
#     def generate_batches(self):
#         n_states = len(self.states)
#         batch_start = np.arange(0, n_states, self.batch_size)
#         indices = np.arange(n_states, dtype=np.int64)
#         np.random.shuffle(indices)
#         batches = [indices[i:i + self.batch_size] for i in batch_start]
#
#         return np.array(self.states), \
#                np.array(self.actions), \
#                np.array(self.probs), \
#                np.array(self.vals), \
#                np.array(self.rewards), \
#                np.array(self.dones), \
#                batches
#
#     def store_memory(self, state, action, probs, vals, reward, done):
#         self.states.append(state)
#         self.actions.append(action)
#         self.probs.append(probs)
#         self.vals.append(vals)
#         self.rewards.append(reward)
#         self.dones.append(done)
#
#     def clear_memory(self):
#         self.states = []
#         self.probs = []
#         self.actions = []
#         self.rewards = []
#         self.dones = []
#         self.vals = []

class PPOMemory:
    def __init__(self, batch_size, sequence_length=5):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size
        self.sequence_length = sequence_length  # Length of the sequence to be sampled

    def generate_batches(self, n_recent=10):
        n_states = len(self.states)
        terminal_indices = [i for i, x in enumerate(self.dones) if x]

        # Generating sequence start indices for terminal states
        sequence_starts = []
        for idx in terminal_indices:
            start = idx - self.sequence_length + 1
            if start < 0:
                start = 0
            sequence_starts.append(start)

        # Add the most recent sequences
        recent_starts = list(range(max(0, n_states - n_recent * self.sequence_length), n_states, self.sequence_length))
        sequence_starts.extend(recent_starts)

        np.random.shuffle(sequence_starts)  # Shuffle the start indices

        # Create batches from the sequence starts
        batches = [sequence_starts[i:i + self.batch_size] for i in range(0, len(sequence_starts), self.batch_size)]

        # Convert the start indices to sequence indices
        sequence_batches = []
        for batch in batches:
            sequence_batch = []
            for start in batch:
                sequence_batch.extend(range(start, min(start + self.sequence_length, n_states)))
            sequence_batches.append(sequence_batch)

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               sequence_batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self,memory=True):
        if memory:
            terminal_indices = [i for i, x in enumerate(self.dones) if x]
            # Keep only the last 4 terminal states and 8 states before each
            keep_indices = set()
            for terminal_index in terminal_indices[-4:]:
                start_index = max(terminal_index - 8, 0)
                keep_indices.update(range(start_index, terminal_index + 1))

            # Update memory lists to only include the required states
            self.states = [self.states[i] for i in sorted(keep_indices)]
            self.probs = [self.probs[i] for i in sorted(keep_indices)]
            self.vals = [self.vals[i] for i in sorted(keep_indices)]
            self.actions = [self.actions[i] for i in sorted(keep_indices)]
            self.rewards = [self.rewards[i] for i in sorted(keep_indices)]
            self.dones = [self.dones[i] for i in sorted(keep_indices)]
        else:
            self.states = []
            self.probs = []
            self.vals = []
            self.actions = []
            self.rewards = []
            self.dones = []

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=625, fc2_dims=256, name='actor_torch_ppo', ver_name='', chkpt_dir='data/SavedModels/PPO'):
        super(ActorNetwork, self).__init__()
        self.name = name + ver_name
        self.checkpoint_file = os.path.join(chkpt_dir, self.name)
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=625, fc2_dims=256, name='critic_torch_ppo', ver_name='',
                 chkpt_dir='data/SavedModels/PPO'):
        super(CriticNetwork, self).__init__()
        self.name = name+ver_name
        self.checkpoint_file = os.path.join(chkpt_dir, self.name)
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, ver_name=''):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha,ver_name=ver_name)
        self.critic = CriticNetwork(input_dims, alpha, ver_name=ver_name)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation, prev_action=None, bias=0.1):
        observation = np.array([observation])
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)

        if prev_action is not None:
            # Get the probabilities from the distribution
            probs = dist.probs.squeeze(0)  # Assuming dist is a Categorical distribution

            # Apply bias towards the previous action
            adjusted_probs = probs.clone()
            adjusted_probs[prev_action] += bias
            adjusted_probs /= adjusted_probs.sum()  # Normalize the probabilities

            # Create a new distribution with the adjusted probabilities for sampling
            adjusted_dist = Categorical(probs=adjusted_probs)
            action = adjusted_dist.sample()
        else:
            # No previous action, sample as usual
            action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self, memory =True):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory(memory)
