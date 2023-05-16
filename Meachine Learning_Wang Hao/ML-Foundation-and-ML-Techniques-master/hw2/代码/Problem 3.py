# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:07:00 2019

@author: qinzhen
"""

import numpy as np

delta=0.05
dvc=10

def f(N):
    return (8 / N * np.log(4 * ((2 * N) ** dvc + 1) / delta)) ** 0.5 - 0.05

n = 1
while(True):
    if(f(n) <= 0):
        break
    else:
        n += 1

print(n)