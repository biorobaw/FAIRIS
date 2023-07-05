import math
import random
from random import uniform

import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from functools import reduce
import xml.etree.ElementTree as ET

def allXall(df1,df2):
  return df1.merge(df2,on='key',sort=False)

def dataFrame(colname,values):
  if isinstance(values, list) or isinstance(values, np.ndarray):
    # print(values)
    return pd.DataFrame({'key': 0, colname: values})
  else:
    return pd.DataFrame({'key': 0, colname: [values]})

def load_maze_random_default(number_obstacles=10,file_name='outmaze.xml'):
	# maze for bio experiments: 
	#		obstacle length = 0.19*sqrt(0.3538) = 0.113 # biology length * distance scale ratio 
	# 		num obstacles: 0, 6, 11, 23
	# maze for robot experiments:
	#		obstacle length 25cm
	#		num obstacles: 0, 10, 20, 30, 40, 50, 60
	# NOTE: min distance between obstacles set to 10cm
	walls = pd.concat([external_walls(), default_random_obsticals(number_obstacles)], ignore_index=True)
	feeders = feeders_maze_default()
	starts  = start_pos_maze_default()
	write_maze_file(walls, feeders, starts,file_name)


def external_walls():
	return pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'],
						data=[
								[-2.0, -3.0, -2.0, 3.0],
								[-2.0, 3.0, 2.0, 3.0],
								[2.0, -3.0, 2.0, 3.0],
								[-2.0, -3.0, 2.0, -3.0]
							 ]
						)

def dummy_walls():
	return pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'])


def feeders_maze_default():
	return pd.DataFrame(
			columns=['id','x','y'],
			data=[
				[int(1), 0.0, 1.5]
			]
		)

def start_pos_maze_default():
	return pd.DataFrame(
			columns=['x','y', 'w'],
			data=[
				[-1.5, -0.5, 0],
				[1.5, -0.5, 0],
				[-1.5, -2.5, 0],
				[1.5, -2.5, 0]
			]
		)

def make_obstical(x,y):
	obstical_length = .75
	theta = uniform(0, 2 * math.pi)
	x1 = x + (obstical_length/2)*(math.cos(theta))
	y1 = y + (obstical_length/2)*(math.sin(theta))
	x2 = x + (obstical_length / 2) * (math.cos(theta+math.pi))
	y2 = y + (obstical_length / 2) * (math.sin(theta+math.pi))
	return [x1,y1,x2,y2]

def default_random_obsticals(number_obsticals):
	x_centers = [-1.5,-.5,.5,1.5]
	y_centers = [-2.5,-1.5,-.5,.5,1.5,2.5]
	not_valid_positions = [(-1.5,-0.5),(1.5,-0.5),(-1.5,-2.5),(1.5,-2.5),(0.0,1.5)]
	possible_positions = []
	for x in x_centers:
		for y in y_centers:
			if not ((x,y) in not_valid_positions):
				possible_positions.append((x,y))

	random_positions = random.sample(possible_positions,number_obsticals)
	obsticals = []
	for p in random_positions:
		obsticals.append(make_obstical(p[0],p[1]))


	return pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2'],
						data=obsticals
	)



def write_maze_file(walls, feeders, starts,file_name):

	world = ET.Element('world')
	startPostions = ET.SubElement(world,'startPositions')
	for index, row in starts.iterrows():
		pos = ET.SubElement(startPostions,'pos',x=str(row['x']),y=str(row['y']),w=str(row['w']))

	for index, row in feeders.iterrows():
		feeder = ET.SubElement(world,'feeder',id=str(int(row['id'])),x=str(row['x']),y=str(row['y']))
	for index, row in walls.iterrows():
		wall = ET.SubElement(world,'wall',x1=str(row['x1']),y1=str(row['y1']),x2=str(row['x2']),y2=str(row['y2']))
	tree = ET.ElementTree(world)
	ET.indent(tree, space="\t", level=0)
	tree.write(file_name,xml_declaration=True)