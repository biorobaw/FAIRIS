import os

from fairis_tools.experiment_tools.paths.path_planning.MazePaths import MazePaths

os.chdir('../../../../simulation/worlds/mazes/Experiment1')
print(os.getcwd())


files = ['WM00.xml', 'WM10.xml','WM20.xml']
# files = ['WM00.xml']
for file in files:
    maze_paths = MazePaths(file)
    maze_paths.calculate_shortest_paths()
    maze_paths.save()
