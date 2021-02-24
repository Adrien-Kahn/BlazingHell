import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import time
import sys

from automata import Automaton
from data_generator import perlin

cmap = ListedColormap(['g', 'r', 'k'])
boundaries = [0, 1.5, 2.5, 4]
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

#  Here we want to create an automata and simulate it several times to see how much the output value disperse


# Setting the parameter of the automaton

c_i = 0.5
c_m = -7

# Create a moisture map

x,y = 50,50
lx = np.linspace(0,5,x,endpoint=False)
ly = np.linspace(0,5,y,endpoint=False)
X,Y = np.meshgrid(lx,ly)
moisture = perlin(X,Y,seed=4) + 0.5


value = []
t = time.time()
n = 1000

for k in range(n):
	auto = Automaton(c_intercept = c_i, c_moisture = c_m, shape = (x,y), firestart = (int(x/2),int(y/2)), moisture = moisture)
	value.append(auto.run())
	sys.stdout.write("\r{:.2%}".format((float(k)/n)))
	sys.stdout.flush()

print('\n')

print('execution time:')
print(time.strftime('%H:%M:%S', time.gmtime(time.time() - t)))

plt.hist(value, bins = 100)


# The conclusion of all this is that output values of the model are essentially distributed in one or more gaussians. So we should find a way to identify which gaussian the output is most likely to belong to and then compute the risk as the distance between the center of that gaussian and the value from the data.