import matplotlib.pyplot as plt
from matplotlib import collections as pycol
from matplotlib import patches

from Simulation.libraries.SimulationLib.Environment import Maze


def make_maze_plot_no_pc(maze_file="Simulation/worlds/mazes/Experiment1/WM00.xml"):
    maze = Maze(maze_file)

    fig, axs = plt.subplots(figsize=(10, 10))

    axs.add_collection(pycol.LineCollection(maze.walls, linewidths=2))

    for point in maze.experiment_starting_location:
        new_crc = patches.Circle((point.x, point.y), radius=.05, color='green')
        axs.add_patch(new_crc)
    for point in maze.habituation_start_location:
        new_crc = patches.Circle((point.x, point.y), radius=.05, color='blue')
        axs.add_patch(new_crc)
    for point in maze.goal_locations:
        new_crc = patches.Circle((point.x, point.y), radius=.05, color='red')
        axs.add_patch(new_crc)

    axs.set_ylim(-3, 3)
    axs.set_xlim(-3, 3)
    axs.margins(0.1)
    file_out_name = maze_file.split('/')[-1].split('.')[0]
    fig.savefig("Simulation/DataCache/" + file_out_name + ".png")

def make_maze_plot_no_pc(maze_file="Simulation/worlds/mazes/Experiment1/WM00.xml"):
    maze = Maze(maze_file)

    fig, axs = plt.subplots(figsize=(10, 10))

    axs.add_collection(pycol.LineCollection(maze.walls, linewidths=2))

    for point in maze.experiment_starting_location:
        new_crc = patches.Circle((point.x, point.y), radius=.05, color='green')
        axs.add_patch(new_crc)
    for point in maze.habituation_start_location:
        new_crc = patches.Circle((point.x, point.y), radius=.05, color='blue')
        axs.add_patch(new_crc)
    for point in maze.goal_locations:
        new_crc = patches.Circle((point.x, point.y), radius=.05, color='red')
        axs.add_patch(new_crc)

    axs.set_ylim(-3, 3)
    axs.set_xlim(-3, 3)
    axs.margins(0.1)
    file_out_name = maze_file.split('/')[-1].split('.')[0]
    fig.savefig("Simulation/DataCache/" + file_out_name + ".png")
