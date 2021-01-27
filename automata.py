
import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from copy import copy
from time import time


def sigmoid(x):
	return 1/(1 + np.exp(-x))

# Chaque cellule de l'automate contient un objet cell qui contient toutes les données nécéssaires au calcul de la fonction de transition
# Pour commencer Cell contient juste state et moisture

# Cell.state vaut :
#1 pour une cellule enflammable
#2 pour une cellule enflammée
#3 pour une cellule consummée

cmap = ListedColormap(['g', 'r', 'k'])
boundaries = [0, 1.5, 2.5, 4]
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

# Une cellule enflammée reste dans cet état pendant c_fuel étapes

c_fuel = 10



class Cell:
	
	def __init__(self, state, moisture):
		self.state = state
		self.moisture = moisture
		self.fuel = c_fuel
		
	def __str__(self):
		return str(self.state)
	
	
# 	Not useful anymore
	def transition(self, neighbors, c_intercept, c_moisture):
		new_c = copy(self)
		if self.state == 1:
			n = len([c for c in neighbors if c.state == 2])
			if n > 0:
				p = sigmoid(n*(c_intercept + c_moisture*self.moisture))
				if np.random.random() < p:
					new_c.state = 2
		if self.state == 2:
			new_c.fuel -= 1
			if new_c.fuel == 0:
				new_c.state = 3
		return new_c
		



class Automaton:
	
	def __init__(self, c_intercept, c_moisture, shape, firestart, moisture):

		self.ii, self.jj = shape

#		builds the matrix that represents the current state of the automata
		mat = np.zeros(shape, dtype = object)
		for i in range(self.ii):
			for j in range(self.jj):
 				mat[i,j] = Cell(1, moisture[i,j])
		mat[firestart].state = 2
		self.matrix = mat
		
		self.c_intercept = c_intercept
		self.c_moisture = c_moisture
		self.time = 0
		self.fire_nb = 1

#		builds a matrix that contains the list of the indexes neighbors of the current index
		self.neighborsMatrix = np.zeros(shape, dtype = object)
		for ci in range(self.ii):
			for cj in range(self.jj):
				self.neighborsMatrix[ci,cj] = [(i,j) for j in range(cj - 1, cj + 2) for i in range(ci - 1, ci + 2) if  i >= 0 and i < self.ii and j >= 0 and j < self.jj and (i != ci or j != cj)]
		
#		builds a set that will contain the indexes of burning cells and their neighbors
		self.cache = set([firestart] + self.neighborsMatrix[firestart])
		
		
# Unused
	def get_neighbors(self, ci, cj):
		return [self.matrix[i,j] for j in range(cj - 1, cj + 2) for i in range(ci - 1, ci + 2) if  i >= 0 and i < self.ii and j >= 0 and j < self.jj and (i != ci or j != cj)]
		
	
	
	def time_step(self):
		
#		we do not copy the matrix but update the current one
#		assuming that cache indeed contains what we want it to, if we do not count the number of neighboring burning cells, we do not need the entire former state to compute the future step of a single cell
		
#		new_cache will contain the indexes of burning cells and their neighbors in the next time step
		new_cache = set([])
		
#		we loop over the indexes in cache
		for index in self.cache:
			c = self.matrix[index]

#			if the cell is not yet burned, we check if it catches on fire
			if c.state == 1:
#				WE DO NOT COUNT THE NUMBER OF NEIGHBORING BURNING CELLS
				p = sigmoid(self.c_intercept + self.c_moisture*c.moisture)
				if np.random.random() < p:
					c.state = 2
					self.fire_nb += 1
					new_cache.update([index] + self.neighborsMatrix[index])
			
# 			if the cell is burning, we decrease fuel and check whether there is still left
			if c.state == 2:
				c.fuel -= 1
				if c.fuel == 0:
					c.state = 3
					self.fire_nb -= 1
				else:
					new_cache.update([index] + self.neighborsMatrix[index])
			
		self.cache = new_cache		
		self.time += 1
				


	def run(self):
	
		t0 = time()
		
		while self.fire_nb > 0:
			self.time_step()
		s = 0
		for i in range(self.ii):
			for j in range(self.jj):
				if self.matrix[i,j].state == 3:
					s += 1

		if __name__ == "__main__":		
			print("execution time : {:.2f}s".format(time() - t0))
			print("execution steps : {}".format(self.time))
			print("number of burned cells : {}".format(s))
			print("percentage of burned cells : {:.2%}".format(s/(self.ii * self.jj)))
	
		return s


	def state_matrix(self):
		sm = np.zeros((self.ii, self.jj))
		for i in range(self.ii):
			for j in range(self.jj):
				sm[i,j] = self.matrix[i,j].state
		return sm
	
	def color_matrix(self):
		sm = self.state_matrix()
		sm[sm == 1] = np.array([0,255,0])
		sm[sm == 2] = np.array([255,0,0])
		sm[sm == 3] = np.array([0,0,0])
		return sm

	def evolution_animation(self):
		fig, ax = plt.subplots()
		im = ax.imshow(self.state_matrix(), animated = True)
		def update(x):
			im.set_data(self.state_matrix())
			self.time_step()
		FuncAnimation(fig, update, interval = 300)


# Tests and visualization

if __name__ == "__main__":
	
	n = 100
	
	x = np.linspace(-1,1,n)
	X,Y = np.meshgrid(x,x)
	
	auto = Automaton(c_intercept = 4, c_moisture = -7, shape = (n,n), firestart = (int(n/2), int(n/2)), moisture = X**2 + Y**2)
	
# 	auto.run()
	
	fig, ax = plt.subplots()
	im = ax.imshow(auto.state_matrix(), cmap = cmap, norm = norm)
	def update(x):
		im.set_array(auto.state_matrix())
		auto.time_step()
	ani = FuncAnimation(fig, update, interval = 100)
	