import numpy as np
import matplotlib.pylab as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
from time import time


def sigmoid(x):
	return 1/(1 + np.exp(-x))

# The wind makes everything more complicated

# Cell.state vaut :
#1 pour une cellule enflammable
#2 pour une cellule enflammée
#3 pour une cellule consummée

cmap = ListedColormap(['g', 'r', 'k'])
boundaries = [0, 1.5, 2.5, 4]
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

# Une cellule enflammée reste dans cet état pendant c_fuel étapes

c_fuel = 5


#    1
#  3   4
#    2
	 
# Builds a matrix that contains the dict of the indexes neighbors of the current index
# Each entry of the dict is 
# This will be called once in the machine object since all automata will have the same shape
# That way we save some time instead of rebuilding it over and over again
def neighbors_matrix(shape):
	ii, jj = shape
	neighborsMatrix = np.zeros(shape, dtype = object)
	for i in range(ii):
		for j in range(jj):
			neighborsMatrix[i,j] = {}
			if i != 0:
				neighborsMatrix[i, j][(i - 1, j)] = 1
			if i != ii - 1:
				neighborsMatrix[i, j][(i + 1, j)] = 2
			if j != 0:
				neighborsMatrix[i, j][(i, j - 1)] = 3
			if j != jj - 1:
				neighborsMatrix[i, j][(i, j + 1)] = 4
	return neighborsMatrix




class Cell:
	
	def __init__(self, state, moisture, vd, windx, windy):
		self.state = state
		
		self.moisture = moisture
		self.vd = vd
		self.windx = windx
		self.windy = windy
				
		self.fuel = c_fuel
		
	def __str__(self):
		return str(self.state)
	



# Just a simple struct-like class to store all the coefficient we need in an efficient way
class Coef:
	def __init__(self, c_intercept, c_moisture, c_vd, c_wind):
		self.intercept = c_intercept
		self.moisture = c_moisture
		self.vd = c_vd
		self.wind = c_wind
	def __str__(self):
		return "c_intercept =\t{:.2f} \nc_moisture =\t{:.2f} \nc_vd =\t\t{:.2f} \nc_wind =\t{:.2f}".format(self.intercept, self.moisture, self.vd, self.wind)
# 	Returns a Coef objects that is a copy of self
	def copy_coef(self):
		coef = Coef(self.intercept, self.moisture, self.vd, self.wind)
		return coef



class Automaton:
	
	def __init__(self, coef, shape, neighborsMatrix, firestart, moisture, vd, windx, windy):

		self.ii, self.jj = shape

#		builds the matrix that represents the current state of the automata
		mat = np.zeros(shape, dtype = object)
		for i in range(self.ii):
			for j in range(self.jj):
 				mat[i,j] = Cell(1, moisture[i,j], vd[i,j], windx, windy)
		mat[firestart].state = 2
		self.matrix = mat
		
		self.neighborsMatrix = neighborsMatrix
		self.coef = coef
		self.time = 0
		self.fire_nb = 1
		
# 		Builds a dict that contains all burning cells and their neighboring burnable cells
# 		Each entry is the coordinate of such a cell
# 		Each value is a list that contains indications on which neighbors is burning:
# 			1 -> upper neighbor (smaller i)
# 			2 -> lower neighbor (bigger i)
# 			3 -> left  neighbor (smaller j)
# 			4 -> right neighbor (bigger j)

		self.cache = {}
		self.cache[firestart] = "fire"
		d = neighborsMatrix[firestart]
		for coor in d:
			self.cache.setdefault(coor, []).append(d[coor])
	
	
	def time_step(self):
				
#		new_cache will replace self.cache in the next step
		new_cache = {}
		
#		we loop over the entries in cache
		for index in self.cache:
			c = self.matrix[index]


# 			if the cell is burning, we decrease fuel and check whether there is still left
			if self.cache[index] == "fire":
				
				# For testing purpose
				if c.state != 2:
					print("Big Problem !")
				
				c.fuel -= 1
				if c.fuel == 0:
					c.state = 3
					self.fire_nb -= 1
				else:
					new_cache[index] = "fire"
					d = self.neighborsMatrix[index]
					for coor in d:
						if new_cache.setdefault(coor, []) != "fire":
							new_cache[coor].append(d[coor])



#			if the cell is not yet burned, we check if it catches on fire
			elif c.state == 1:
				
#				This is where the fun begins
				
# 				Here is the inverted diagram that indicates where the fire is based on the number
#    2
#  4   3
#    1
				
# 				b will help us get out of here faster if ignition has already happened
				b = True				

				s = self.coef.intercept + self.coef.moisture * c.moisture + self.coef.vd * c.vd
				
# 				For each individual neighboring fire, we check for ignition
				l = self.cache[index]
				
# 				A cell is burning below (higher i) the current cell
				if b:
					if 1 in l:
						p = sigmoid(s + self.coef.wind * c.windy)
						if np.random.random() < p:
							
# 							Changing the state to burning
							c.state = 2
							self.fire_nb += 1
# 							Adding the coordinates to new_cache
							new_cache[index] = "fire"
							d = self.neighborsMatrix[index]
							for coor in d:
								if new_cache.setdefault(coor, []) != "fire":
									new_cache[coor].append(d[coor])
# 	 						Not checking whether other neighboring cells lead to ignition
							b = False


# 				A cell is burning above (lower i) the current cell
				if b:
					if 2 in l:
						p = sigmoid(s - self.coef.wind * c.windy)
						if np.random.random() < p:
							c.state = 2
							self.fire_nb += 1
							new_cache[index] = "fire"
							d = self.neighborsMatrix[index]
							for coor in d:
								if new_cache.setdefault(coor, []) != "fire":
									new_cache[coor].append(d[coor])
							b = False
					

# 				A cell is burning to the right (higher j) of the current cell
				if b:
					if 3 in l:
						p = sigmoid(s - self.coef.wind * c.windx)
						if np.random.random() < p:
							c.state = 2
							self.fire_nb += 1
							new_cache[index] = "fire"
							d = self.neighborsMatrix[index]
							for coor in d:
								if new_cache.setdefault(coor, []) != "fire":
									new_cache[coor].append(d[coor])
							b = False


# 				A cell is burning to the left (lower j) of the current cell
				if b:
					if 4 in l:
						p = sigmoid(s + self.coef.wind * c.windx)
						if np.random.random() < p:
							c.state = 2
							self.fire_nb += 1
							new_cache[index] = "fire"
							d = self.neighborsMatrix[index]
							for coor in d:
								if new_cache.setdefault(coor, []) != "fire":
									new_cache[coor].append(d[coor])
							b = False

			
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
	
	
	def final_state(self):
		while self.fire_nb > 0:
			self.time_step()
		return self.state_matrix()


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



# Tests and visualization

if __name__ == "__main__":
		
	n = 100
	nshape = (n,n)
	fs = (int(n/2), int(n/2))
	
	nM = neighbors_matrix((n,n))
	coef = Coef(-3, 0, 0, 1)
	
	x = np.linspace(-1,1,n)
	X,Y = np.meshgrid(x,x)
	
# 	moisture = X**2 + Y**2
# 	vd = (X + 1)**2 / 4

	moisture = np.full(nshape, 0)
	vd = np.full(nshape, 0)
	
	windx = -3
	windy = -2
	
	auto = Automaton(coef, nshape, nM, fs, moisture, vd, windx, windy)
	
	
	fig, ax = plt.subplots()
	im = ax.imshow(auto.state_matrix(), cmap = cmap, norm = norm)
	
	def update(x):
		im.set_array(auto.state_matrix())
		auto.time_step()
	
	ani = FuncAnimation(fig, update, interval = 100)
	
	"""
	f = r"c://Users/adrie/Desktop/animation.gif" 
	writergif = animation.PillowWriter(fps=30) 
	ani.save(f, writer=writergif)
	"""


"""
	
	fs = auto.final_state()
	plt.imshow(fs)
	print("Burnt cells: {}".format(len(fs[fs == 3])))
	
"""

