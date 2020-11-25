import automata
import pandas as pd
import numpy as np

class machine:
	
	def __init__(self, data):
		self.data = data


a = np.array([[1,2],[3,4]])
b = np.array([[10,20],[30,40]])

r = np.random.randn(2,3)
df = pd.DataFrame(r, columns = list('xyz'))

dff = pd.DataFrame(np.array([[a,b],[1,2]]), columns = list('xy'))