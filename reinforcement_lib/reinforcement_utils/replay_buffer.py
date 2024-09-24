import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.mem_cntr = 0
        self.final_cntr = 0
        self.prob_sequence = 0.5
        self.state_memory = np.zeros((self.mem_size, *self.input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *self.input_shape))
        self.action_memory = np.zeros((self.mem_size, self.n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.final_states = []

    def store_transition(self, state, action, reward, state_, done):
        self.check_replay_buffer()
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        if reward >= 0:
            self.final_states.append(index)
            self.final_cntr += 1
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def strat_sample_buffer(self, batch_size):

        if np.random.random() < self.prob_sequence and self.final_cntr >= batch_size:
            # Sample a sequence
            final_batch_index = np.random.choice(self.final_states, size=1)[0]
            # Check if we have enough previous states to form a sequence
            if final_batch_index - batch_size + 1 >= 0:
                batch = [i for i in range(final_batch_index - batch_size + 1, final_batch_index + 1)]
                states = self.state_memory[batch]
                states_ = self.new_state_memory[batch]
                actions = self.action_memory[batch]
                rewards = self.reward_memory[batch]
                dones = self.terminal_memory[batch]
                return states, actions, rewards, states_, dones

        # If we're here, then either we're not sampling a sequence or we don't have enough states for a sequence
        return self.sample_buffer(batch_size)

    def increase_prop_sequence(self):
        self.prob_sequence+=.01
        self.prob_sequence = min(self.prob_sequence, .95)

    def check_replay_buffer(self):
        if self.mem_cntr >self.mem_size:
            print("Replay Memory Full. Clearing Buffer")
            self.mem_cntr = 0
            self.final_cntr = 0
            self.prob_sequence = 0.5
            self.state_memory = np.zeros((self.mem_size, *self.input_shape))
            self.new_state_memory = np.zeros((self.mem_size, *self.input_shape))
            self.action_memory = np.zeros((self.mem_size, self.n_actions))
            self.reward_memory = np.zeros(self.mem_size)
            self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
            self.final_states = []