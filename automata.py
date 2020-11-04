
import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
from copy import copy


def sigmoid(x):
	return 1/(1 + np.exp(-x))



# Chaque cellule de l'automate contient un objet cell qui contient toutes les données nécéssaires au calcul de la fonction de transition
# Pour commencer Cell contient juste state et moisture
# moisture est affecté d'un coefficient c_m dans la régression linéaire et l'intercept d'un coefficient c_i

c_i = 1
c_m = -1

# Cell.state vaut :
#1 pour une cellule enflammable
#2 pour une cellule enflammée
#3 pour une cellule consummée

# Une cellule enflammée reste dans cet état pendant c_fuel étapes

c_fuel = 4



class Cell:
	
	def __init__(self, state, moisture):
		self.state = state
		self.moisture = moisture
		self.fuel = c_fuel
		
	def __str__(self):
		return str(self.state)
	
	def transition(self, neighbors):
		new_c = copy(self)
		if self.state == 1:
			n = len([c for c in neighbors if c.state == 2])
			if n > 0:
				p = sigmoid(n*(c_i + c_m*self.moisture))
				if np.random.random() < p:
					new_c.state = 2
		if self.state == 2:
			new_c.fuel -= 1
			if new_c.fuel == 0:
				new_c.state = 3
		return new_c
		


# La fonction de transition locale prend en argument une cellule et ses voisins, et renvoie la cellule à l'étape de temps suivante

class Automaton:
	
	def __init__(self, initial_matrix):
		self.matrix = initial_matrix
		self.ii, self.jj = self.matrix.shape
		self.time = 0
	
	def get_neighbors(self, ci, cj):
		return [self.matrix[i,j] for j in range(cj - 1, cj + 2) for i in range(ci - 1, ci + 2) if  i >= 0 and i < self.ii and j >= 0 and j < self.jj and (i != ci or j != cj)]
	
	def time_step(self):
		new_matrix = np.zeros((self.ii, self.jj), dtype = object)
		for i in range(self.ii):
			for j in range(self.jj):
				new_matrix[i,j] = self.matrix[i,j].transition(self.get_neighbors(i,j))
		self.matrix = new_matrix
		self.time += 1

	def state_matrix(self):
		sm = np.zeros((self.ii, self.jj))
		for i in range(self.ii):
			for j in range(self.jj):
				sm[i,j] = self.matrix[i,j].state
		return sm

	def evolution_animation(self):
		fig, ax = plt.subplots()
		im = ax.imshow(self.state_matrix(), animated = True)
		def update(x):
			im.set_array(self.state_matrix())
			self.time_step()
		FuncAnimation(fig, update, interval = 300)


imax = 20
jmax = 20

initial_matrix = np.zeros((imax, jmax), dtype = object)

for i in range(imax):
	for j in range(jmax):
		initial_matrix[i,j] = Cell(1, 1)

initial_matrix[10, 10] = Cell(2, 1)

auto = Automaton(initial_matrix)

fig, ax = plt.subplots()
im = ax.imshow(auto.state_matrix(), animated = True)
def update(x):
	im.set_array(auto.state_matrix())
	auto.time_step()
ani = FuncAnimation(fig, update, interval = 300)
