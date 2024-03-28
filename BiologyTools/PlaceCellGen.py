import os
import pickle

import numpy as np

os.chdir("..")
print(os.getcwd())

"""ExperimentSupervisor controller."""
from BiologyTools.PlaceCellLibrary import PlaceCellNetwork

xs = [[-3,-1.8, -0.6, 0.6, 1.8, 3],
      [-3,-2.25,-1.5,-0.75,0,0.75,1.5,2.25,3],
      [-3, -2.45, -1.91, -1.36, -0.82, -0.23, 0.23, 0.82, 1.36, 1.91, 2.45, 3],
      np.linspace(-3,3,num=25)]

ys = [[-3,-1.8, -0.6, 0.6, 1.8, 3],
      [-3,-2.25,-1.5,-0.75,0,0.75,1.5,2.25,3],
      [-3, -2.45, -1.91, -1.36, -0.82, -0.23, 0.23, 0.82, 1.36, 1.91, 2.45, 3],
      np.linspace(-3,3,num=25)]

pc_network_name = ['uniform_6','uniform_9','uniform_12','uniform_25']

rs = [(1.2)*(.75),(.75)*(.75),(6/11)*(.75),0.3]

# xs =  [-3,-1.8, -0.6, 0.6, 1.8, 3]
# ys =  [-3,-1.8, -0.6, 0.6, 1.8, 3]

# xs =  [-3,-2.25,-1.5,-0.75,0,0.75,1.5,2.25,3]
# ys =  [-3,-2.25,-1.5,-0.75,0,0.75,1.5,2.25,3]

# xs =  [-3, -2.45, -1.91, -1.36, -0.82, -0.23, 0.23, 0.82, 1.36, 1.91, 2.45, 3]
# ys =  [-3, -2.45, -1.91, -1.36, -0.82, -0.23, 0.23, 0.82, 1.36, 1.91, 2.45, 3]

for i in range(4):
    experiment_pc_network = PlaceCellNetwork()
    for x in xs[i]:
        for y in ys[i]:
            experiment_pc_network.add_pc_to_network(x, y, radius=rs[i])

    with open("Simulation/GeneratedPCNetworks/" + pc_network_name[i], 'wb') as pc_file:
        pickle.dump(experiment_pc_network, pc_file)