from trueAutomata import Automaton
from trueAutomata import neighbors_matrix
from trueAutomata import Coef
import pandas as pd
import numpy as np
import random as rd
from time import time
import os
import matplotlib.pyplot as plt

import ray

# We now will have to ponder the question of firestart

# To start, we take for firestart the point with the highest value in the burned area array




# This seed controls pretty much everything else
npseed = 1
np.random.seed(npseed)

# This seed controls which data indices are chosen in minibatches
rd.seed(0)


# Builds a dataframe by fetching data from the folder indicated in path
# path: relative path to the folder containing all fire instances
#  n: optionnal parameter to limit the number of entries fetched
def create_dataframe(path, n = -1):
	
	k = 0
	df = pd.DataFrame(columns = ['entry number', 'value', 'moisture', 'vd', 'windx', 'windy', 'firestart', 'shape', 'neighborsMatrix'])
	
	for filename in os.scandir(path):
				
		if filename.name[0:9] == 'Australia':
	
			print("Loading " + filename.name)
			
			number = [int(i) for i in filename.name.split("_") if i.isdigit()][0]
			
			for entry in os.scandir(filename.path):
				
				prefix = path + '/' + filename.name + '/'
				
				if entry.name == 'vegetation_density.csv':
					vd = np.loadtxt(prefix + entry.name, delimiter=",", skiprows=1)
				elif entry.name == 'burned_area.csv':
					burned_area = np.loadtxt(prefix + entry.name, delimiter=",", skiprows=1)
					value = np.zeros(burned_area.shape)
					value[burned_area > 0.5] = 3
					value[burned_area <= 0.5] = 1
				elif entry.name == 'humidity.csv':
					moisture = np.loadtxt(prefix + entry.name, delimiter=",", skiprows=1)
			
			# Get the argmax of burned area for firestart
			firestart = np.unravel_index(value.argmax(), value.shape)
			
			# Store the shape and neighborsMatrix
			shape = value.shape
			nM = neighbors_matrix(shape)
			
			df.loc[k] = [number, value, moisture, vd, 0, 0, firestart, shape, nM]
			k += 1
			
			if k == n:
				break
	
# 	Fetching the winds

	print("\nLoading winds")

	windfile = open(path + "/processed_wind.txt")
	winddata = windfile.read()
	entries = winddata.split("\n")
	
	entries = [s.split() for s in entries]
	
	for i in range(len(df.index)):
		
		for line in entries:
						
			if int(line[0]) == df.iloc[i]['entry number']:
				
				df.at[i, 'windx'] = float(line[1])
				df.at[i, 'windy'] = float(line[2])
				
				break
		
		if df.at[i, 'windx'] == 0:
			print("Failed to fetch wind for entry {}".format(df.iloc[i]['entry number']))

	return df


# A function that output the square of the amount of different cells in a and b
# It is used to define the cost of an output of the model

def matdiff(a,b):
	return len(a[a != b])**2


# We write a version of gradient that is not a method of machine but a function that takes 
# all the required arguments
# This way we can easily make it remote

def fgradient(coef, shape, nM, firestart, moisture, vd, windx, windy, value, h, m):		

	g_i = 0
	g_m = 0
	g_vd = 0
	g_w = 0
		
	coef_i = coef.copy_coef()
	coef_i.intercept += h
	
	coef_m = coef.copy_coef()
	coef_m.moisture += h
	
	coef_vd = coef.copy_coef()
	coef_vd.vd += h
	
	coef_w = coef.copy_coef()
	coef_w.wind += h
		
	for i in range(m):

		auto_0 = Automaton(coef, shape, nM, firestart, moisture, vd, windx, windy)
		c_0 = matdiff(auto_0.final_state(), value)

			
		auto_i = Automaton(coef_i, shape, nM, firestart, moisture, vd, windx, windy)
		c_i = matdiff(auto_i.final_state(), value)
		g_i += (c_i - c_0)/h
				
		auto_m = Automaton(coef_m, shape, nM, firestart, moisture, vd, windx, windy)
		c_m = matdiff(auto_m.final_state(), value)
		g_m += (c_m - c_0)/h
			
		auto_vd = Automaton(coef_vd, shape, nM, firestart, moisture, vd, windx, windy)
		c_vd = matdiff(auto_vd.final_state(), value)
		g_vd += (c_vd - c_0)/h

		auto_w = Automaton(coef_w, shape, nM, firestart, moisture, vd, windx, windy)
		c_w = matdiff(auto_w.final_state(), value)
		g_w += (c_w - c_0)/h
		
	return g_i / m, g_m / m, g_vd / m, g_w / m


remote_fgradient = ray.remote(fgradient)


# Same for the remote version of cost
def fcost(coef, shape, neighborsMatrix, firestart, moisture, vd, windx, windy, value, m):
	
	c = 0
	
	for k in range(m):
		auto = Automaton(coef, shape, neighborsMatrix, firestart, moisture, vd, windx, windy)
		c += matdiff(auto.final_state(), value)
	
	return c/m


remote_fcost = ray.remote(fcost)



class machine:
	
	def __init__(self, data, mb_size, coef_ini, h, learning_rate, remote = False, cluster = False):
		
		if remote and (not ray.is_initialized()):
			if cluster:
				ray.init(address = "auto")
			else:
				ray.init()
		
		self.data = data
		self.mb_size = mb_size
		self.coef = coef_ini
		self.h = h
		self.learning_rate = learning_rate
		self.remote = remote

	
	def __str__(self):
		return str(self.coef)
	
	
# 	Computes the quadratic cost on the k-th data point averaged over m computations
	
	def cost(self, k, m):

		value = self.data.iloc[k]['value']		
		moisture = self.data.iloc[k]['moisture']
		vd = self.data.iloc[k]['vd']
		windx = self.data.iloc[k]['windx']
		windy = self.data.iloc[k]['windy']
		firestart = self.data.iloc[k]['firestart']
		shape = self.data.iloc[k]['shape']
		nM = self.data.iloc[k]['neighborsMatrix']
		
		c = 0
		
		for i in range(m):
			auto = Automaton(self.coef, shape, nM, firestart, moisture, vd, windx, windy)
			c += matdiff(auto.final_state(), value)			
		
		return c/m


# 	Computes the average quadratic cost over all data points
# 	For each data point, the cost is averaged over m computations
	
	def fullcost(self, m = 1):
		if self.remote:
			return self.remote_fullcost(m)
		else:
			return self.normal_fullcost(m)
	

	def normal_fullcost(self, m):
		c = 0
		n = len(self.data)
		for k in range(n):
			c += self.cost(k, m)
			print(self.cost(k,m))
		return c/n
	
	
	def remote_fullcost(self, m):
		n = len(self.data)
		futures = [remote_fcost.remote(self.coef, self.data.iloc[k]['shape'], self.data.iloc[k]['neighborsMatrix'], self.data.iloc[k]['firestart'], self.data.iloc[k]['moisture'], self.data.iloc[k]['vd'], self.data.iloc[k]['windx'], self.data.iloc[k]['windy'], self.data.iloc[k]['value'], m) for k in range(n)]
		cost_list = ray.get(futures)
		return np.average(cost_list)

	
# 	Computes the approximation of the gradient for the instance of data at index k
# 	The computed value is averaged over m computations
	
	def gradient(self, k, m):
		
# 		Fetching all the necessary data
		value = self.data.iloc[k]['value']		
		moisture = self.data.iloc[k]['moisture']
		vd = self.data.iloc[k]['vd']
		windx = self.data.iloc[k]['windx']
		windy = self.data.iloc[k]['windy']
		firestart = self.data.iloc[k]['firestart']
		shape = self.data.iloc[k]['shape']
		nM = self.data.iloc[k]['neighborsMatrix']
		
# 		Initializing the values that will store the gradient
		g_i = 0
		g_m = 0
		g_vd = 0
		g_w = 0
		
# 		Building the Coef objects
		coef_i = self.coef.copy_coef()
		coef_i.intercept += self.h
		
		coef_m = self.coef.copy_coef()
		coef_m.moisture += self.h
		
		coef_vd = self.coef.copy_coef()
		coef_vd.vd += self.h
		
		coef_w = self.coef.copy_coef()
		coef_w.wind += self.h
		
		for i in range(m):

#			On calcule d'abord la prédiction sur le paramètre actuel			
			auto_0 = Automaton(self.coef, shape, nM, firestart, moisture, vd, windx, windy)
			c_0 = matdiff(auto_0.final_state(), value)
			
#			Puis avec chacun des paramètres augmenté de self.h
#			On peut ensuite calculer la dérivée partielle par rapport à chaque paramètre et actualiser le gradient
	
#			c_intercept + h d'abord
			auto_i = Automaton(coef_i, shape, nM, firestart, moisture, vd, windx, windy)
			c_i = matdiff(auto_i.final_state(), value)
			g_i += (c_i - c_0)/self.h
				
#			c_moisture + h ensuite
			auto_m = Automaton(coef_m, shape, nM, firestart, moisture, vd, windx, windy)
			c_m = matdiff(auto_m.final_state(), value)
			g_m += (c_m - c_0)/self.h
			
			auto_vd = Automaton(coef_vd, shape, nM, firestart, moisture, vd, windx, windy)
			c_vd = matdiff(auto_vd.final_state(), value)
			g_vd += (c_vd - c_0)/self.h

			auto_w = Automaton(coef_w, shape, nM, firestart, moisture, vd, windx, windy)
			c_w = matdiff(auto_w.final_state(), value)
			g_w += (c_w - c_0)/self.h
		
		return g_i / m, g_m / m, g_vd / m, g_w / m

#	Performs a step of the gradient descent
#	The gradient computed on each data point is averaged over m computations
	def learn_step(self, m):
		if self.remote:
			self.remote_learn_step(m)
		else:
			self.normal_learn_step(m)
	
	
	def normal_learn_step(self, m):
		
# 		First we get a subset of mb_size element of data
# 		We randomly select rows, and each row might get selected more than once
		
		n = len(self.data)
		l = [k for k in range(n)]
		ll = rd.choices(l, k = self.mb_size)
		
# 		The point of all this is to compute the gradient over the minibatch so we initiate a gradient and then average over all rows of the minibatch
		
		grad_intercept = 0
		grad_moisture = 0
		grad_vd = 0
		grad_wind = 0
		
		for k in ll:			
			g_i, g_m, g_vd, g_w = self.gradient(k, m)
			
			grad_intercept += g_i
			grad_moisture += g_m
			grad_vd += g_vd
			grad_wind += g_w
			
# 		Then we use the gradient by normalizing it and adding it to the parameter
			
		grad_intercept = (grad_intercept/self.mb_size)*self.learning_rate
		grad_moisture = (grad_moisture/self.mb_size)*self.learning_rate
		grad_vd = (grad_vd/self.mb_size)*self.learning_rate
		grad_wind = (grad_wind/self.mb_size)*self.learning_rate
		
		self.coef.intercept -= grad_intercept
		self.coef.moisture -= grad_moisture
		self.coef.vd -= grad_vd
		self.coef.wind -= grad_wind

	
	def remote_learn_step(self, m):
		
# 		Ray was already initialized in the constructor if ray.remote
# 		so we don't need to take care of it here
		
		n = len(self.data)
		l = [k for k in range(n)]
		ll = rd.choices(l, k = self.mb_size)
		
		grad_intercept = 0
		grad_moisture = 0
		grad_vd = 0
		grad_wind = 0
		
# 		Instead of computing gradient on a data point and incrementing the gradients directly,
# 		we ask for a remote computation of all the gradients and we increment once the
# 		computation is over
		
		futures = [remote_fgradient.remote(self.coef, self.data.iloc[k]['shape'], self.data.iloc[k]['neighborsMatrix'], self.data.iloc[k]['firestart'], self.data.iloc[k]['moisture'], self.data.iloc[k]['vd'], self.data.iloc[k]['windx'], self.data.iloc[k]['windy'], self.data.iloc[k]['value'], self.h, m) for k in ll]
		grad_list = ray.get(futures)
		
		for g_i, g_m, g_vd, g_w in grad_list:
			grad_intercept += g_i
			grad_moisture += g_m
			grad_vd += g_vd
			grad_wind += g_w		
		
		grad_intercept = (grad_intercept/self.mb_size)*self.learning_rate
		grad_moisture = (grad_moisture/self.mb_size)*self.learning_rate
		grad_vd = (grad_vd/self.mb_size)*self.learning_rate
		grad_wind = (grad_wind/self.mb_size)*self.learning_rate
		
		self.coef.intercept -= grad_intercept
		self.coef.moisture -= grad_moisture
		self.coef.vd -= grad_vd
		self.coef.wind -= grad_wind
		
	
	def learning(self, n, m):
		for k in range(n):
			self.learn_step(m)
	
	def predict(self, firestart, moisture, vd, windx, windy):
		auto = Automaton(self.coef, self.shape, self.neighborsMatrix, firestart, moisture, vd, windx, windy)
		return auto.final_state()







print("\nFetching database...\n")

bigdata = create_dataframe("processing_result", 10)

print("\nData fetched successfully")

coef = Coef(0, 0, 0, 0)

daneel = machine(bigdata, 30, coef, h = 0.1, learning_rate = 0.00003, remote = True, cluster = True)

print("\n\nMachine built: \n")

print("Initial parameters:\n")
print(daneel)
print()

t1 = time()

# for cost computation
m1 = 1

# for gradient computation
m2 = 1

def fmat():

	xn = 3
	yn = 3
	
	x = np.linspace(0, 1, xn)
	y = np.linspace(0, 1, yn)
	
	mat = np.zeros((xn,yn))
	
	for i in range(xn):
		for j in range(yn):
			daneel.coef.moisture = x[i]
			daneel.coef.wind = y[j]
			mat[i,j] = daneel.fullcost(m1)
	
	return mat


def flin(x, xn):
	
	l = []	
	
	for i in range(xn):
		daneel.coef.wind = x[i]
		c = daneel.fullcost(1)
		print(np.log(c))
		l.append(np.log(c))
	
	return l



xn = 10
x = np.linspace(0, 10, xn)

l = flin(x, xn)

plt.plot(x, l)
plt.show()

"""

print("\nInitial log cost:\t{:.2f}\n".format(np.log(daneel.fullcost(m1))))

print("Initiating learning phase...\n")

for k in range(100):
	daneel.learn_step(m2)
	print("Log cost at step {}:\t{:.2f}\n".format(k, np.log(daneel.fullcost(m1))))
	print("Parameters at step {}:".format(k))
	print(daneel)
	print("\n")

print("Learning phase time: {:.2f}s\n\n".format(time() - t1))

"""



if ray.is_initialized():
	ray.shutdown()

print("\n\nComputation over\n")
