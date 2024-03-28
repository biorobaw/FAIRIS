class ExperimentLogger:
    def __init__(self, maze_file, pc_file):
        self.maze_file = maze_file
        self.pc_dist = pc_file
        self.score_history = []
        self.episodes = []
        self.path_length_history = []

    def log_episode(self, episode, score,path_length):
        self.episodes.append(episode)
        self.score_history.append(score)
        self.path_length_history.append(path_length)

class EpisodeLogger:
    def __init__(self,start_x,start_y):
        self.path_xs = [start_x]
        self.path_ys = [start_y]
        self.path_length = 0

    def log_step(self, robot_x, robot_y):
        self.path_xs.append(robot_x)
        self.path_ys.append(robot_y)
        self.path_length += 1
