import os
import pickle

os.chdir("..")

"""ExperimentSupervisor controller."""
from BiologyTools.PlaceCellLibrary import PlaceCellNetwork

xs =  [-3,-2.25,-1.5,-0.75,0,0.75,1.5,2.25,3]
ys =  [-3,-2.25,-1.5,-0.75,0,0.75,1.5,2.25,3]

pc_network_name = 'uniform_test'

experiment_pc_network = PlaceCellNetwork()
for x in xs:
    for y in ys:
        experiment_pc_network.add_pc_to_network(x, y, radius=.5)

with open("Simulation/GeneratedPCNetworks/" + pc_network_name, 'wb') as pc_file:
    pickle.dump(experiment_pc_network, pc_file)