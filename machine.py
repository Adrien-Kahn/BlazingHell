from automata import Automaton
from data_generator import data
import pandas as pd
import numpy as np
import random as rd
from time import time
import matplotlib.pyplot as plt

import ray

# Pour commencer, pour pas trop se prendre la tête, on va fixer firestart constamment à x_max/2, y_max/2


# The evalution of the averaged cost of the dataset leads to very different values over the same instance, even for a dataset with n = 100. As such, we recommend averaging over many iteration though it will be very costly.

# Initial cost: 229536.74
# Initial cost 2: 162438.16
# Initial cost 3: 187141.67
# Initial cost 4: 148423.07
# Initial cost 5: 226678.29
# Initial cost 5: 153856.01
# Initial cost 6: 186287.94


# This seed controls pretty much everything else
npseed = 1
np.random.seed(npseed)

# This seed controls which data indices are chosen in minibatches
rd.seed(0)


# A function that output the square of the amount of different cells in a and b
# It is used to define the cost of an output of the model

def matdiff(a,b):
	return len(a[a != b])**2


# We write a version of gradient that is not a method of machine but a function that takes 
# all the required arguments
# This way we can easily make it remote

def fgradient(moisture, value, c_intercept, c_moisture, h, m):

	x,y = moisture.shape
	
	g_i = 0
	g_m = 0
	
	for k in range(m):
		
#		On calcule d'abord la prédiction sur le paramètre actuel			
		auto_0 = Automaton(c_intercept, c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
		c_0 = matdiff(auto_0.final_state(), value)
		
#		Puis avec chacun des paramètres augmenté de self.h
#		On peut ensuite calculer la dérivée partielle par rapport à chaque paramètre et actualiser le gradient

#		c_intercept + h d'abord
		auto_i = Automaton(c_intercept + h, c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
		c_i = matdiff(auto_i.final_state(), value)
		g_i += (c_i - c_0)/h
			
#		c_moisture + h ensuite
		auto_m = Automaton(c_intercept, c_moisture + h, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
		c_m = matdiff(auto_m.final_state(), value)
		g_m += (c_m - c_0)/h
		
	return g_i / m, g_m / m


remote_fgradient = ray.remote(fgradient)


# Same for the remote version of cost
def fcost(moisture, value, c_intercept, c_moisture, m):
	
	x,y = moisture.shape
	c = 0
	
	for k in range(m):
		auto = Automaton(c_intercept, c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
		c += matdiff(auto.final_state(), value)
	
	return c/m


remote_fcost = ray.remote(fcost)



class machine:
	
	def __init__(self, data, mb_size, c_intercept = 0, c_moisture = 0, h = 1, learning_rate = 0.00001, remote = False, cluster = False):
		
		if remote and (not ray.is_initialized()):
			if cluster:
				ray.init(address = "auto")
			else:
				ray.init()
		
		self.data = data
		self.mb_size = mb_size
		self.c_intercept = c_intercept
		self.c_moisture = c_moisture
		self.h = h
		self.learning_rate = learning_rate
		self.remote = remote
	
	
	def __str__(self):
		return "c_intercept = {:.2f} \nc_moisture = {:.2f} \n".format(self.c_intercept, self.c_moisture)
	
	
# 	Computes the quadratic cost on the k-th data point averaged over m computations
	
	def cost(self, k, m):
		
		moisture = self.data.iloc[k]['moisture']
		value = self.data.iloc[k]['value']
		x,y = moisture.shape
		
		c = 0
		
		for i in range(m):
			auto = Automaton(self.c_intercept, self.c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
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
		futures = [remote_fcost.remote(self.data.iloc[k]['moisture'], self.data.iloc[k]['value'], self.c_intercept, self.c_moisture, m) for k in range(n)]
		cost_list = ray.get(futures)
		return np.average(cost_list)

	
# 	Computes the approximation of the gradient for the instance of data at index k
# 	The computed value is averaged over m computations
	
	def gradient(self, k, m):
		
		moisture = self.data.iloc[k]['moisture']
		value = self.data.iloc[k]['value']
		
		x,y = moisture.shape
		
		g_i = 0
		g_m = 0
		
		for i in range(m):

#			On calcule d'abord la prédiction sur le paramètre actuel			
			auto_0 = Automaton(self.c_intercept, self.c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
			c_0 = matdiff(auto_0.final_state(), value)
			
#			Puis avec chacun des paramètres augmenté de self.h
#			On peut ensuite calculer la dérivée partielle par rapport à chaque paramètre et actualiser le gradient
	
#			c_intercept + h d'abord
			auto_i = Automaton(self.c_intercept + self.h, self.c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
			c_i = matdiff(auto_i.final_state(), value)
			g_i += (c_i - c_0)/self.h
				
#			c_moisture + h ensuite
			auto_m = Automaton(self.c_intercept, self.c_moisture + self.h, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
			c_m = matdiff(auto_m.final_state(), value)
			g_m += (c_m - c_0)/self.h
		
		return g_i / m, g_m / m

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
		
		for k in ll:			
			g_i, g_m = self.gradient(k, m)
			
			grad_intercept += g_i
			grad_moisture += g_m
			
# 		Then we use the gradient by normalizing it and adding it to the parameter
			
		grad_intercept = (grad_intercept/self.mb_size)*self.learning_rate
		grad_moisture = (grad_moisture/self.mb_size)*self.learning_rate
		
		self.c_intercept -= grad_intercept
		self.c_moisture -= grad_moisture

	
	def remote_learn_step(self, m):
		
# 		Ray was already initialized in the constructor if ray.remote
# 		so we don't need to take care of it here
		
		n = len(self.data)
		l = [k for k in range(n)]
		ll = rd.choices(l, k = self.mb_size)
		
		grad_intercept = 0
		grad_moisture = 0
		
# 		Instead of computing gradient on a data point and incrementing the gradients directly,
# 		we ask for a remote computation of all the gradients and we increment once the
# 		computation is over
		
		futures = [remote_fgradient.remote(self.data.iloc[k]['moisture'], self.data.iloc[k]['value'], self.c_intercept, self.c_moisture, self.h, m) for k in ll]
		grad_list = ray.get(futures)
		
		for g_i, g_m in grad_list:
			grad_intercept += g_i
			grad_moisture += g_m		
		
		grad_intercept = (grad_intercept/self.mb_size)*self.learning_rate
		grad_moisture = (grad_moisture/self.mb_size)*self.learning_rate
		
		self.c_intercept -= grad_intercept
		self.c_moisture -= grad_moisture
		
	
	def learning(self, n, m):
		for k in range(n):
			self.learn_step(m)
	
	def predict(self, moisture, firestart = (25,25)):
		auto = Automaton(self.c_intercept, self.c_moisture, moisture.shape, firestart, moisture)
		return auto.final_state()

print("\nBuilding database...\n")

bigdata = data(100, 1, -7, shape = (50,50), firestart = (25,25), revert_seed = npseed)

print("Data generated\n")


daneel = machine(bigdata, mb_size = 30, c_intercept = 0.54, c_moisture = -6.31, h = 0.1, learning_rate = 0.0000003, remote = True, cluster = True)

print("\n\nMachine built: \n")

print("Initial parameters:\n")
print(daneel)



"""
# With m = 10 we get a full cost with precision around 1%
a = 20

t0 = time()

for k in range(6):
	print("Initial cost averaged {} times:\t{}".format(a, daneel.fullcost(a)))

print("\nCost computation time: {:.2f}s\n\n".format(time() - t0))

print("\n")

# At b = 30, we start having reasonably small variation (around 20% at most)
b = 30
for k in range(10):
	x,y = daneel.gradient(0,b)
	print("Gradient at point 0 averaged {} times:\t{}\t{}".format(b, x, y))

"""

t1 = time()

# for cost computation
m1 = 20

# for gradient computation
m2 = 40


c_i, c_m = daneel.c_intercept, daneel.c_moisture
daneel.c_intercept, daneel.c_moisture =	1, -7
print("\nTarger log cost:\t{:.2f}\n".format(np.log(daneel.fullcost(m1))))
daneel.c_intercept, daneel.c_moisture = c_i, c_m


print("\nInitial log cost:\t{:.2f}\n".format(np.log(daneel.fullcost(m1))))

print("Initiating learning phase...\n")

for k in range(100):
	daneel.learn_step(m2)
	print("Log cost at step {}:\t{:.2f}\n".format(k, np.log(daneel.fullcost(m1))))
	print("Parameters at step {}:".format(k))
	print(daneel)
	print("\n")

if ray.is_initialized():
	ray.shutdown()

print("Computation over\n")
print("Learning phase time: {:.2f}s\n\n".format(time() - t1))

