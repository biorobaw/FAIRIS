import numpy as np

class PPOMemory:
    def __init__(self, batch_size, sequence_length=20):
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
            for terminal_index in terminal_indices[-8:]:
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
