# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:34:46 2021

@author: adrie
"""
import pandas as pd
import numpy as np
import os



def create_dataframe(_path): #_path = l'adresse du dossier contenant tous les sous-dossiers 
	dictionnaire = {}
	for filename in os.scandir(_path): # filename = l'ensemble des sous-dossiers
		print("Loading " + filename.name)
		for entry in os.scandir(filename.path): # entry = l'ensemble des dossiers csv
			if entry.name=='vegetation_dansity.csv':
				vegetation_dansity=np.loadtxt(_path + '/' + filename.name + '/' + entry.name,delimiter=",", skiprows=1)
			elif entry.name=='burned_area.csv':
				burned_area=np.loadtxt(_path + '/' + filename.name + '/' + entry.name,delimiter=",", skiprows=1)
			elif entry.name=='humidity.csv':
				humidity=np.loadtxt(_path + '/' + filename.name + '/' + entry.name,delimiter=",", skiprows=1)
		dictionnaire[filename.name]=(vegetation_dansity, burned_area, humidity)
	return (pd.DataFrame.from_dict(dictionnaire, orient = 'index'))

