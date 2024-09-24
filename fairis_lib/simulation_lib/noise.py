import numpy as np


# Define noise functions
def add_gaussian_noise(x, y, mean=0, std_dev=0.1):
    noise_x = np.random.normal(mean, std_dev)
    noise_y = np.random.normal(mean, std_dev)
    x_noisy = np.clip(x + noise_x, -3, 3)
    y_noisy = np.clip(y + noise_y, -3, 3)
    return x_noisy, y_noisy


def add_uniform_noise(x, y, low=-0.1, high=0.1):
    noise_x = np.random.uniform(low, high)
    noise_y = np.random.uniform(low, high)
    x_noisy = np.clip(x + noise_x, -3, 3)
    y_noisy = np.clip(y + noise_y, -3, 3)
    return x_noisy, y_noisy


def add_salt_and_pepper_noise(x, y, prob=0.05):
    if np.random.rand() < prob:
        x = np.random.choice([np.inf, -np.inf])
    if np.random.rand() < prob:
        y = np.random.choice([np.inf, -np.inf])
    x_noisy = np.clip(x, -3, 3)
    y_noisy = np.clip(y, -3, 3)
    return x_noisy, y_noisy


def add_exponential_noise(x, y, scale=0.1):
    noise_x = np.random.exponential(scale)
    noise_y = np.random.exponential(scale)
    x_noisy = np.clip(x + noise_x, -3, 3)
    y_noisy = np.clip(y + noise_y, -3, 3)
    return x_noisy, y_noisy


# Initialize random walk parameters globally
random_walk_state = {'walk_x': 0, 'walk_y': 0, 'steps': 0}


def add_random_walk_noise(x, y, step_size=0.01):
    global random_walk_state
    random_walk_state['steps'] += 1
    if random_walk_state['steps'] > 10:
        random_walk_state = {'walk_x': 0, 'walk_y': 0, 'steps': 0}
    random_walk_state['walk_x'] += np.random.normal(0, step_size)
    random_walk_state['walk_y'] += np.random.normal(0, step_size)
    x_noisy = np.clip(x + random_walk_state['walk_x'], -3, 3)
    y_noisy = np.clip(y + random_walk_state['walk_y'], -3, 3)
    return x_noisy, y_noisy


# Function to apply noise and report various noise strengths
def apply_noise(x, y, noise_type='gaussian', level=1):
    noise_functions = {
        'gaussian': [lambda x, y: add_gaussian_noise(x, y, std_dev=0.1),
                     lambda x, y: add_gaussian_noise(x, y, std_dev=0.5),
                     lambda x, y: add_gaussian_noise(x, y, std_dev=1.0)],

        'uniform': [lambda x, y: add_uniform_noise(x, y, low=-0.1, high=0.1),
                    lambda x, y: add_uniform_noise(x, y, low=-0.5, high=0.5),
                    lambda x, y: add_uniform_noise(x, y, low=-1.0, high=1.0)],

        'salt_and_pepper': [lambda x, y: add_salt_and_pepper_noise(x, y, prob=0.05),
                            lambda x, y: add_salt_and_pepper_noise(x, y, prob=0.1),
                            lambda x, y: add_salt_and_pepper_noise(x, y, prob=0.2)],

        'exponential': [lambda x, y: add_exponential_noise(x, y, scale=0.1),
                        lambda x, y: add_exponential_noise(x, y, scale=0.5),
                        lambda x, y: add_exponential_noise(x, y, scale=1.0)],

        'random_walk': [lambda x, y: add_random_walk_noise(x, y, step_size=0.01),
                        lambda x, y: add_random_walk_noise(x, y, step_size=0.05),
                        lambda x, y: add_random_walk_noise(x, y, step_size=0.1)]
    }

    noise_function = noise_functions[noise_type][level - 1]
    x_noisy, y_noisy = noise_function(x, y)

    return x_noisy, y_noisy