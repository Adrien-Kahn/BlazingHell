from trueAutomata import Automaton
from trueAutomata import neighbors_matrix
from trueAutomata import Coef
import pandas as pd
import numpy as np
import random as rd
from time import time
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import ray



# This seed controls pretty much everything else
npseed = 1
np.random.seed(npseed)

# This seed controls which data indices are chosen in minibatches
rd.seed(0)


# Scales the input matrix data to the shape (out_x, out_y)
def regrid(data, out_x, out_y):
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)

    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))

    return interpolating_function((xv, yv))



# Builds a dataframe by fetching data from the folder indicated in path
# path: relative path to the folder containing all fire instances
# compression_factor: the factor by which to compress the dimensions of the arrays
# n: optional parameter to limit the number of entries fetched
def create_dataframe(path, compression_factor, n = -1):
	
	cf = compression_factor
	k = 0
	df = pd.DataFrame(columns = ['entry number', 'value', 'moisture', 'vd', 'windx', 'windy', 'firestart', 'shape', 'neighborsMatrix'])
	
	for filename in os.scandir(path):
				
		if filename.name[0:9] == 'Australia':
	
			print("Loading " + filename.name)
			
			number = [int(i) for i in filename.name.split("_") if i.isdigit()][0]
			
			for entry in os.scandir(filename.path):
				
				prefix = path + '/' + filename.name + '/'
				
				if entry.name == 'vegetation_density.csv':
					vegetation_density = np.loadtxt(prefix + entry.name, delimiter=",", skiprows=1)
					x,y = vegetation_density.shape
					vd = regrid(vegetation_density, x//cf, y//cf)
					
				elif entry.name == 'burned_area.csv':
					burned_area = np.loadtxt(prefix + entry.name, delimiter=",", skiprows=1)
					x,y = burned_area.shape
					ba = regrid(burned_area, x//cf, y//cf)
					value = np.zeros(ba.shape)
					value[ba > 0.5] = 3
					value[ba <= 0.5] = 1
					
				elif entry.name == 'humidity.csv':
					humidity = np.loadtxt(prefix + entry.name, delimiter=",", skiprows=1)
					x,y = humidity.shape
					moisture = regrid(humidity, x//cf, y//cf)
			
			# Get the argmax of burned area for firestart
			firestart = np.unravel_index(ba.argmax(), value.shape)
			
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


def success_rate(a, b):
	return a[a == b].size / a.size

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


def ftest(coef, shape, neighborsMatrix, firestart, moisture, vd, windx, windy, real_value, m):
	
	c = 0
	
	for k in range(m):
		auto = Automaton(coef, shape, neighborsMatrix, firestart, moisture, vd, windx, windy)
		c += success_rate(auto.final_state(), real_value)
	
	return c/m

remote_test = ray.remote(ftest)



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
	
	
# 	Computes the success rate of the model on the test_data dataset by averaging the success rate of each entry m times
	def test(self, test_data, m):
		n = len(test_data)
		futures = [remote_test.remote(self.coef, test_data.iloc[k]['shape'], test_data.iloc[k]['neighborsMatrix'], test_data.iloc[k]['firestart'], test_data.iloc[k]['moisture'], test_data.iloc[k]['vd'], test_data.iloc[k]['windx'], test_data.iloc[k]['windy'], test_data.iloc[k]['value'], m) for k in range(n)]
		srate_list = ray.get(futures)
		return np.average(srate_list)





print("\nFetching database...\n")

# Gets all data entries and compresses their size by a factor of 5
bigdata = create_dataframe("../processing_result", 5)

# Splitting the database between training data and testing data
train_size = 67

train_data = bigdata[:train_size]
test_data = bigdata[train_size:]


print("\nFetched {} entries".format(len(bigdata)))
print("{} entries for training".format(len(train_data)))
print("{} entries for testing\n\n".format(len(test_data)))


coef = Coef(-19.5, -0.7, 38.6, -0.25)

daneel = machine(train_data, 50, coef, h = 0.1, learning_rate = 0.000001, remote = True, cluster = True)

print("Machine built: \n")

print("Initial parameters:\n")
print(daneel)
print()

print("\nInitial success rate over test data:\t{:.2%}\n".format(daneel.test(test_data, 10)))

t1 = time()

# for cost computation
m1 = 10

# for gradient computation
m2 = 30

# for success rate computation
m3 = 10

# Plots the projection of the cost in the plane of two of the four coordinates
def fmat():

	xn = 10
	yn = 10
	
	x = np.linspace(30, 50, xn)
	y = np.linspace(-1, 5, yn)
	
	lx = []
	ly = []
	lc = []

	for i in range(xn):
		for j in range(yn):
			print(i,j)
			daneel.coef.vd = x[i]
			daneel.coef.wind = y[j]
			lx.append(x[i])
			ly.append(y[j])
			lc.append(np.log(daneel.fullcost(m1)))
	
	plt.scatter(lx, ly, s = 1000, c = lc, cmap = 'viridis')
	plt.colorbar()
	plt.xlabel("c_vd")
	plt.ylabel("c_wind")
	plt.legend()
	plt.show()
	print(lx)
	print(ly)
	print(lc)


# Plots the projection of the cost along the axis of one coordinate
def flin():
	
	xn = 100
	x = np.linspace(-50, 50, xn)
	l = []
	
	for i in range(xn):
		daneel.coef.wind = x[i]
		c = daneel.fullcost(m1)
		l.append(np.log(c))
	
	plt.plot(x, l)
	plt.xlabel("c_wind")
	plt.ylabel("log cost")
	plt.title(str(daneel))
	plt.legend()
	plt.show()



# Does learning steps and prints the evolution of the parameters and of the cost
def learningTest():
	
	li = [daneel.coef.intercept]
	lm = [daneel.coef.moisture]
	lv = [daneel.coef.vd]
	lw = [daneel.coef.wind]
	lc = [np.log(daneel.fullcost(m1))]
	
	print("\nInitial log cost:\t{:.2f}\n".format(lc[0]))
	print("Initiating learning phase...\n")
	
	for k in range(10000):
	
		tk = time()
		daneel.learn_step(m2)
		print("Learning step time:\t\t{:.2f}s".format(time() - tk))
		
		tk = time()
		logc = np.log(daneel.fullcost(m1))
		print("Cost computation time:\t\t{:.2f}s".format(time() - tk))
		
		tk = time()
		srate = daneel.test(test_data, m3)
		print("Success rate computation time:\t{:.2f}s\n".format(time() - tk))

		li.append(daneel.coef.intercept)
		lm.append(daneel.coef.moisture)
		lv.append(daneel.coef.vd)
		lw.append(daneel.coef.wind)
		lc.append(logc)
		
		print("Log cost at step {}:\t{:.2f}".format(k, logc))
		print("Success rate at step {}:\t{:.2%}\n".format(k, srate))
		print("Parameters at step {}:".format(k))
		print(daneel)
	
		if k%5 == 0:	
			print("List of c_intercept:")
			print(li)
			print("List of c_moisture:")
			print(lm)
			print("List of c_vd:")
			print(lv)
			print("List of c_wind:")
			print(lw)
			print("List of log cost:")
			print(lc)
	
		print("\n")
		
	print("Learning phase time: {:.2f}s\n\n".format(time() - t1))




"""
for k in range(20):
	c = Coef(100*np.random.random() - 50, 100*np.random.random() - 50, 100*np.random.random() - 50, 100*np.random.random() - 50)
	daneel.coef = c
	print(daneel)
	print("Success rate over test data:\t{:.2%}\n\n".format(daneel.test(test_data, 10)))	
"""


#fmat()

learningTest()




if ray.is_initialized():
	ray.shutdown()

print("\n\nComputation over\n")
