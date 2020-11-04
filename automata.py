
import numpy as np

# Chaque cellule de l'automate contient un objet cell qui contient toutes les données nécéssaires au calcul de la fonction de transition
#Pour commencer Cell contient juste state et moisture

class Cell:
	
	def __init__(self, state):
		self.state = state
		
	def get_state(self):
		return self.state
	
	def set_state(self, new_state):
		self.state = new_state


# La fonction de transition locale, contenu dans la variable transition, prend en argument une cellule et ses voisins, et renvoie l'état de la cellule à l'étape de temps suivante

class Automaton:
	
	def __init__(self, initial_state, transition):
		self.matrix = initial_state
		self.ii, self.jj = self.matrix.shape
		self.transition = transition
		self.time = 0
	
	def time_step(self):
		new_matrix = np.zeros((self.ii, self.jj))
		for i in range(self.ii):
			for j in range(self.jj):
				self.transition(i,j)
	
	def get_neighbors(self, i, j)