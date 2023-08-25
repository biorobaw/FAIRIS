from matplotlib import collections as pycol
from matplotlib import patches
import matplotlib.pyplot as plt
import os
os.chdir("../..")

from Simulation.libraries.Environment import Maze

maze_file = "Simulation/worlds/mazes/Experiment1/WM00.xml"

maze = Maze(maze_file)

fig, axs = plt.subplots(figsize=(10,10))

axs.add_collection(pycol.LineCollection(maze.walls,linewidths=2))

for point in maze.experiment_starting_location:
    new_crc = patches.Circle((point.x, point.y), radius=.05,color='green')
    axs.add_patch(new_crc)
for point in maze.habituation_start_location:
    new_crc = patches.Circle((point.x, point.y), radius=.05,color='blue')
    axs.add_patch(new_crc)
for point in maze.goal_locations:
    new_crc = patches.Circle((point.x, point.y), radius=.05, color='red')
    axs.add_patch(new_crc)

axs.set_ylim(-3, 3)
axs.set_xlim(-3, 3)
axs.margins(0.1)

fig.savefig("Simulation/DataCache/WM00.png")
