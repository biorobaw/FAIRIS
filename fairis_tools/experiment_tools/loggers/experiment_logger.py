class ExperimentLogger:
    def __init__(self, maze_file=None, pc_file=None,old_data=None):
        if old_data == None:
            self.maze_file = maze_file
            self.pc_dist = pc_file
            self.score_history = []
            self.episodes = []
            self.path_length_history = []
            self.optimality_ratio_history = []
        else:
            self.maze_file = old_data.maze_file
            self.pc_dist = old_data.pc_dist
            self.score_history = old_data.score_history
            self.episodes = old_data.episodes
            self.path_length_history = old_data.path_length_history
            self.optimality_ratio_history = []
    def log_episode(self, episode, score,path_length):
        self.episodes.append(episode)
        self.score_history.append(score)
        self.path_length_history.append(path_length)

    def log_from_csv(self,data):
        for episode in data:
            episode_logger = EpisodeLogger(episode[0][0], episode[0][1])
            for step in episode[1:]:
                episode_logger.log_step(step[0],step[1])
            self.log_episode(episode_logger,0,len(episode))
    def log_optimality_ratio_of_paths(self,episode_data):
        self.optimality_ratio_history.append(episode_data)

class EvaluationLogger:
    def __init__(self, maze_file=None, pc_file=None,old_data=None,noise_type=None,noise_level = 0):
        if old_data == None:
            self.maze_file = maze_file
            self.pc_dist = pc_file
            self.score_history = []
            self.episodes = []
            self.path_length_history = []
            self.optimality_ratio_history = []
            self.noise_type = noise_type
            self.noise_level = noise_level
        else:
            self.maze_file = old_data.maze_file
            self.pc_dist = old_data.pc_dist
            self.score_history = old_data.score_history
            self.episodes = old_data.episodes
            self.path_length_history = old_data.path_length_history
            self.optimality_ratio_history = []
            self.noise_type = old_data.noise_type
            self.noise_level = old_data.noise_level
    def log_episode(self, episode, score,path_length):
        self.episodes.append(episode)
        self.score_history.append(score)
        self.path_length_history.append(path_length)

    def log_from_csv(self,data):
        for episode in data:
            episode_logger = EpisodeLogger(episode[0][0], episode[0][1])
            for step in episode[1:]:
                episode_logger.log_step(step[0],step[1])
            self.log_episode(episode_logger,0,len(episode))
    def log_optimality_ratio_of_paths(self,episode_data):
        self.optimality_ratio_history.append(episode_data)

class EpisodeLogger:
    def __init__(self,start_x,start_y):
        self.path_xs = [start_x]
        self.path_ys = [start_y]
        self.path_length = 0

    def log_step(self, robot_x, robot_y):
        self.path_xs.append(robot_x)
        self.path_ys.append(robot_y)
        self.path_length += 1

class ShortestPath:
    def __init__(self,maze_file,starting_points,path_lengths):
        self.maze_file = maze_file
        self.starting_points = starting_points
        self.path_lengths = {
            starting_points[0] : path_lengths[0],
            starting_points[1] : path_lengths[1],
            starting_points[2] : path_lengths[2],
            starting_points[3] : path_lengths[3],
            starting_points[4] : path_lengths[4],
            starting_points[5] : path_lengths[5],
            starting_points[6] : path_lengths[6],
            starting_points[7] : path_lengths[7]
        }