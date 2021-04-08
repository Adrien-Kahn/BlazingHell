# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:34:46 2021

@author: adrie
"""
import pandas as pd
import numpy as np
import os



def create_dataframe(path): # path: relative path to the folder containing all fire instances
	
	k = 0
	df = pd.DataFrame(columns = ['entry number', 'burned area', 'vegetation density', 'humidity'])
	
	for filename in os.scandir(path):
		print("Loading " + filename.name)
		
		number = [int(i) for i in filename.name.split("_") if i.isdigit()][0]
		
		for entry in os.scandir(filename.path):
			
			prefix = path + '/' + filename.name + '/'
			
			if entry.name == 'vegetation_dansity.csv':
				vegetation_density = np.loadtxt(prefix + entry.name, delimiter=",", skiprows=1)
			elif entry.name == 'burned_area.csv':
				burned_area = np.loadtxt(prefix + entry.name, delimiter=",", skiprows=1)
			elif entry.name == 'humidity.csv':
				humidity = np.loadtxt(prefix + entry.name, delimiter=",", skiprows=1)
			
		df.loc[k] = [number, burned_area, vegetation_density, humidity]
		k += 1

	return df

