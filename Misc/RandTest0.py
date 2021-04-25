# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:43:32 2021

@author: adrie
"""
import numpy as np

def f(seed):
	np.random.seed(seed)
	print(np.random.random())

# print("from 0: {}".format(np.random.random()))
