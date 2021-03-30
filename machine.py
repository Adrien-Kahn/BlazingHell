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

def fgradient(moisture, value, c_intercept, c_moisture, h):

	x,y = moisture.shape

# 		On calcule d'abord la prédiction sur le paramètre actuel			
	auto_0 = Automaton(c_intercept, c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
	c_0 = matdiff(auto_0.final_state(), value)
		
# 		Puis avec chacun des paramètres augmenté de self.h
# 		On peut ensuite calculer la dérivée partielle par rapport à chaque paramètre et actualiser le gradient

# 		c_intercept + h d'abord
	auto_i = Automaton(c_intercept + h, c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
	c_i = matdiff(auto_i.final_state(), value)
	g_i = (c_i - c_0)/h
			
# 		c_moisture + h ensuite
	auto_m = Automaton(c_intercept, c_moisture + h, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
	c_m = matdiff(auto_m.final_state(), value)
	g_m = (c_m - c_0)/h
		
	return g_i, g_m


remote_fgradient = ray.remote(fgradient)



class machine:
	
	def __init__(self, data, mb_size, c_intercept = -10, c_moisture = 20, h = 1, learning_rate = 0.000001, remote = False):
		
		if remote and (not ray.is_initialized()):
			ray.init(address = "auto")
		
		self.data = data
		self.mb_size = mb_size
		self.c_intercept = c_intercept
		self.c_moisture = c_moisture
		self.h = h
		self.learning_rate = learning_rate
		self.remote = remote
	
	
	def __str__(self):
		return "c_intercept = {} \nc_moisture = {} \n".format(self.c_intercept, self.c_moisture)
	
	
# 	Computes the quadratic cost of the prediction on the instance k of data
	
	def cost(self, k):
		
		moisture = self.data.iloc[k]['moisture']
		value = self.data.iloc[k]['value']
		
		x,y = moisture.shape

		auto = Automaton(self.c_intercept, self.c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
		
		return matdiff(auto.final_state(), value)
	
	
# 	Computes the average quadratic cost over all instances of data

	def fullcost(self):
		c = 0
		n = len(self.data)
		for k in range(n):
			c += self.cost(k)
			print(self.cost(k))
		return c/n

	
# 	Computes the approximation of the gradient for the instance of data at index k
	
	def gradient(self, k):
		
		moisture = self.data.iloc[k]['moisture']
		value = self.data.iloc[k]['value']
		
		x,y = moisture.shape

# 		On calcule d'abord la prédiction sur le paramètre actuel			
		auto_0 = Automaton(self.c_intercept, self.c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
		c_0 = matdiff(auto_0.final_state(), value)
		
# 		Puis avec chacun des paramètres augmenté de self.h
# 		On peut ensuite calculer la dérivée partielle par rapport à chaque paramètre et actualiser le gradient

# 		c_intercept + h d'abord
		auto_i = Automaton(self.c_intercept + self.h, self.c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
		c_i = matdiff(auto_i.final_state(), value)
		g_i = (c_i - c_0)/self.h
			
# 		c_moisture + h ensuite
		auto_m = Automaton(self.c_intercept, self.c_moisture + self.h, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
		c_m = matdiff(auto_m.final_state(), value)
		g_m = (c_m - c_0)/self.h
		
		return g_i, g_m

	
	def learn_step(self):
		if self.remote:
			self.remote_learn_step()
		else:
			self.normal_learn_step()
	
	
	def normal_learn_step(self):
		
# 		First we get a subset of mb_size element of data
# 		We randomly select rows, and each row might get selected more than once
		
		n = len(self.data)
		l = [k for k in range(n)]
		ll = rd.choices(l, k = self.mb_size)
		
# 		The point of all this is to compute the gradient over the minibatch so we initiate a gradient and then average over all rows of the minibatch
		
		grad_intercept = 0
		grad_moisture = 0
		
		for k in ll:			
			g_i, g_m = self.gradient(k)
			
			grad_intercept += g_i
			grad_moisture += g_m
			
# 		Then we use the gradient by normalizing it and adding it to the parameter
			
		grad_intercept = (grad_intercept/self.mb_size)*self.learning_rate
		grad_moisture = (grad_moisture/self.mb_size)*self.learning_rate
		
		self.c_intercept -= grad_intercept
		self.c_moisture -= grad_moisture

	
	def remote_learn_step(self):
		
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
		
		futures = [remote_fgradient.remote(self.data.iloc[k]['moisture'], self.data.iloc[k]['value'], self.c_intercept, self.c_moisture, self.h) for k in ll]
		grad_list = ray.get(futures)
		
		for g_i, g_m in grad_list:
			grad_intercept += g_i
			grad_moisture += g_m		
		
		grad_intercept = (grad_intercept/self.mb_size)*self.learning_rate
		grad_moisture = (grad_moisture/self.mb_size)*self.learning_rate
		
		self.c_intercept -= grad_intercept
		self.c_moisture -= grad_moisture
		
	
	def learning(self,n):
		for k in range(n):
			self.learn_step()
	
	def predict(self, moisture, firestart = (25,25)):
		auto = Automaton(self.c_intercept, self.c_moisture, moisture.shape, firestart, moisture)
		return auto.final_state()

print("\nBuilding database...\n")

bigdata = data(100, 1, -7, shape = (50,50), firestart = (25,25), revert_seed = npseed)

print("Data generated\n")


daneel = machine(bigdata, mb_size = 30, remote = True)


# print("Initial cost: {}".format(daneel.fullcost()))
# print("Initial cost 2: {}".format(daneel.fullcost()))
# print("Initial cost 3: {}".format(daneel.fullcost()))
# print("Initial cost 4: {}".format(daneel.fullcost()))
# print("Initial cost 5: {}".format(daneel.fullcost()))
# print("Initial cost 5: {}".format(daneel.fullcost()))
# print("Initial cost 6: {}".format(daneel.fullcost()))

print(daneel)

t0 = time()

for k in range(300):
	daneel.learn_step()
# 	print("Cost: {}".format(daneel.fullcost()))
	print(daneel)

if ray.is_initialized():
	ray.shutdown()

print("Computation over\n")
print("Learning phase time: {:.2f}s\n\n".format(time() - t0)

