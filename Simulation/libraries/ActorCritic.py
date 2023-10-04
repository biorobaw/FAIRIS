import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Define the Q-network


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Actor-Critic model
class ActorCritic:
    def __init__(self, input_size, hidden_size, output_size):
        self.actor = QNetwork(input_size, hidden_size, output_size)
        self.critic = QNetwork(input_size, hidden_size, 1)  # Critic outputs a single value

        # Initialize other parameters
        self.epsilon_initial = 0.9  # Initial exploration rate
        self.epsilon_decay = 0.05  # Rate at which epsilon decays over time
        self.min_epsilon = 0.1  # Minimum exploration rate
        self.current_episode = 0  # Initialize current episode

        # Initialize actor and critic optimizers with weight decay
        weight_decay = 1e-5  # Adjust the weight decay value as needed
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=0.001)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=0.001)

    def select_action(self, state, prev_action, possible_actions):
        q_values = self.actor(state)

        # Apply biases for the previous action and impossible actions
        q_values += 1 * torch.eye(q_values.shape[-1])[prev_action]  # Bias for previous action
        q_values += .9 * torch.eye(q_values.shape[-1])[(prev_action + 1) % 8]
        q_values += .9 * torch.eye(q_values.shape[-1])[(prev_action - 1) % 8]
        q_values += .8* torch.eye(q_values.shape[-1])[(prev_action + 2) % 8]
        q_values += .8 * torch.eye(q_values.shape[-1])[(prev_action - 2) % 8]

        # Convert possible_actions to a PyTorch tensor if it's a list
        if isinstance(possible_actions, list):
            possible_actions = torch.tensor(possible_actions)

        # Create a mask tensor of the same size as q_values
        mask = torch.zeros_like(q_values)
        mask[possible_actions] = 1.0

        # Set Q-values for actions that are not possible to zero
        q_values *= mask

        # Apply temperature scaling for exploration (higher temperature increases exploration)
        epsilon = max(self.epsilon_initial - self.epsilon_decay * self.current_episode, self.min_epsilon)
        if torch.rand(1).item() < epsilon:
            # Apply softmax to the Q-values to get action probabilities
            action_probs = F.softmax(q_values, dim=-1)

            # Handle invalid action probabilities
            invalid_prob_mask = torch.isnan(action_probs) | torch.isinf(action_probs) | (action_probs < 0)
            if invalid_prob_mask.any():
                # Replace invalid values with uniform probabilities among possible actions
                num_valid_actions = possible_actions.size(0)
                valid_probs = torch.ones_like(action_probs) / num_valid_actions
                action_probs[invalid_prob_mask] = valid_probs[invalid_prob_mask]

            action = torch.multinomial(action_probs, 1).item()
            if action not in possible_actions:
                action = random.choice(possible_actions.tolist())

        else:
            action = torch.argmax(q_values).item()
            if action not in possible_actions:
                action =  random.choice(possible_actions.tolist())

        return action

    def start_new_episode(self):
        # Increment the episode counter and reset epsilon for the new episode
        self.current_episode += 1

    def update(self, state, action, reward, next_state, done):
        # Compute the Q-value error (TD error)
        critic_value = self.critic(state)
        next_critic_value = self.critic(next_state)
        target = reward + (1 - done) * 0.99 * next_critic_value

        # Compute the policy loss (actor loss)
        action_probs = torch.softmax(self.actor(state), dim=-1)
        selected_action_prob = action_probs[action]
        actor_loss = -torch.log(selected_action_prob) * (target - critic_value.detach())

        # Compute the critic loss
        critic_loss = nn.MSELoss()(critic_value, target.detach())

        # Backpropagate and apply gradient clipping
        actor_loss.backward()
        critic_loss.backward()

        max_grad_norm = 1.0  # Adjust the maximum gradient norm as needed
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)

        # Update the networks
        self.optimizer_actor.step()
        self.optimizer_critic.step()

    def save_model(self, actor_path, critic_path):
        # Save both actor and critic models to specified paths
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        # Load actor and critic models from specified paths
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

    def test_model(self, state, prev_action, possible_actions):
        # Test the loaded model by selecting an action from the list of possible actions without exploration
        self.actor.eval()  # Set the actor network to evaluation mode
        with torch.no_grad():
            q_values = self.actor(state)
            # Apply biases for the previous action and impossible actions
            q_values += 100 * torch.eye(q_values.shape[-1])[prev_action]
            # Create a mask tensor for possible actions
            mask = torch.zeros_like(q_values)
            mask[possible_actions] = 1.0

            # Set Q-values for actions that are not possible to a large negative value
            q_values -= 1e8 * (1 - mask)

            # Choose the action index with the highest Q-value among possible actions
            while True:
                action = torch.argmax(q_values).item()
                if action in possible_actions:
                    break
                else:
                    q_values[action] = 0.0
        self.actor.train()  # Set the actor network back to training mode
        return action
