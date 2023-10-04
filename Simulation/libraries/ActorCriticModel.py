import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def select_action(logits, possible_actions_mask, previous_action=None, bias_factor=1.2,n_actions=8):
    # Apply the action mask: Set impossible actions to a very large negative number
    masked_logits = logits + (-1e9 * (1 - possible_actions_mask))

    # Bias towards previous action, if provided
    if previous_action is not None:
        bias = np.ones_like(masked_logits.numpy())  # Creates an array of ones with the same shape as logits
        bias[0][previous_action] = bias_factor
        masked_logits = tf.multiply(masked_logits, bias)  # Multiply element-wise

    # Compute action probabilities considering only possible actions
    action_prob = tf.nn.softmax(masked_logits)

    # Sample the next action based on the biased probabilities
    action = np.random.choice(n_actions, p=np.squeeze(action_prob.numpy()))

    return action

class ActorCriticModel(keras.Model):
    def __init__(self, n_actions):
        super(ActorCriticModel, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu', kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.policy_logits = layers.Dense(n_actions, kernel_initializer='he_normal')
        self.values = layers.Dense(1)
        self.huber_loss = tf.keras.losses.Huber()

    def call(self, state):
        x = self.dense1(state)
        x = self.bn1(x)
        logits = self.policy_logits(x)
        values = self.values(x)
        return logits, values

    def compute_loss(self, action_prob, values, rewards):
        advantage = rewards - values
        policy_loss = -tf.math.log(action_prob) * advantage
        value_loss = self.huber_loss(values, rewards)
        return policy_loss + value_loss

    def save_model(self, path):
        self.save_weights(path)

    def load_model(self, path):
        self.load_weights(path)

    def test(self, state):
        state = state.reshape([1, -1])
        logits, _ = self(state)
        action_prob = tf.nn.softmax(logits)
        action = np.argmax(action_prob, axis=1)[0].numpy()
        return action