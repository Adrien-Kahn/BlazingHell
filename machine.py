from automata import Automaton
import pandas as pd
import numpy as np
import random as rd

# Pour commencer, pour pas trop se prendre la tête, on va fixer firestart constamment à x_max/2, y_max/2

class machine:
	
	def __init__(self, data, mb_size, c_intercept = 0, c_moisture = 0, h = 0.01, learning_rate = 0.1):
		self.mb_size = mb_size
		self.c_intercept = c_intercept
		self.c_moisture = c_moisture
		self.h = h
		self.learning_rate = learning_rate
	
	def learn_step(self):
		
# 		First we get a subset of mb_size element of data
# 		We randomly select rows, and each row might get selected more than once
		
		n = len(self.data)
		l = [k for k in range(n)]
		ll = rd.choices(l, self.mb_size)
		
# 		The point of all this is to compute the gradient over the minibatch so we initiate a gradient and then average over all rows of the minibatch
		
		grad_intercept = 0
		grad_moisture = 0
		
		for k in ll:
			moisture = self.data.iloc[k]['moisture']
			value = self.data.iloc[k]['value']
			
# 			On calcule d'abord la prédiction sur le paramètre actuel
			
			x,y = moisture.shape
			auto_0 = Automaton(self.c_intercept, self.c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
			c_0 = (auto_0.run() - value)**2
			
# 			Puis avec chacun des paramètres augmenté de self.h
# 			On peut ensuite calculer la dérivée partielle par rapport à chaque paramètre et actualiser le gradient

# 			c_intercept + h d'abord

			auto_i = Automaton(self.c_intercept + self.h, self.c_moisture, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
			c_i = (auto_i.run() - value)**2
			
			g_i = (c_i - c_0)/self.h
			grad_intercept += g_i
			
# 			c_moisture + h ensuite

			auto_m = Automaton(self.c_intercept, self.c_moisture + self.h, shape = (x,y), firestart = (int(x/2), int(y/2)), moisture = moisture)
			c_m = (auto_m.run() - value)**2
			
			g_m = (c_m - c_0)/self.h
			grad_moisture += g_m
			
# 		Il reste ensuite à utiliser ce gradient
			


a = np.array([[1,2],[3,4]])
b = np.array([[10,20],[30,40]])

r = np.random.randn(2,3)
df = pd.DataFrame(r, columns = list('xyz'))

dff = pd.DataFrame(np.array([[a,b],[1,2]]), columns = list('xy'))