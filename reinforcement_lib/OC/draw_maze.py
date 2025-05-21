import sys
sys.path.append("/home/b/brendon45/FAIRIS/")

import matplotlib.pyplot as plt
from fairis_lib.simulation_lib.environment import Maze


def draw_maze(filename, walls, goal, starting):
    plt.clf()
    fig, ax = plt.subplots()

    for wall in walls:
        ax.plot(wall[0], wall[1], 'black')

    ax.scatter(starting[0], starting[1], c='red')
    ax.scatter(goal[0], goal[1], c='g')

    ax.relim()
    ax.autoscale_view()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    mazes_file = ["fourrooms_obsv2"]

    for maze in mazes_file:
        maze_f = f"/home/b/brendon45/oc_tests/mazes/{maze}.xml"
        m_temp = Maze(maze_f)
        walls, goal, subgoals, starting = m_temp.get_plot_data()
        maze_fn = f"/home/b/brendon45/oc_tests/{maze}.png"
        draw_maze(maze_fn, walls, goal, starting)

