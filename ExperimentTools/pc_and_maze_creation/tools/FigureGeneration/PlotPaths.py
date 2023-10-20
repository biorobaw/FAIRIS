from ExperimentTools.pc_and_maze_creation.tools.path_planning.wavefront_planner import *

os.chdir('../../../../Simulation')
print(os.getcwd())

shortest_path_dir = 'worlds/mazes/Experiment1/Paths/'
# files = ['WM00', 'WM10','WM20']
files = ['WM00']
for file in files:
    with open(shortest_path_dir+file+'_paths.pkl','rb') as data:
        path_data = pickle.load(data)
    # maze = Maze('/worlds/mazes/Experiment1/'+file+'.xml')
    # fig, ax = maze.get_maze_figure()
    for path in path_data.paths:
        for p in path.path:
            print(p)
