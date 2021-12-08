import nashpy as nash
import numpy as np

A = np.array([[100,35],[175,50]])
B = np.array([[100,175],[35,50]])
gm = nash.Game(A,B)
print(gm)
