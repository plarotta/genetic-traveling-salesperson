import numpy as np
from matplotlib import pyplot as plt

a = np.array([[np.cos(val), np.sin(val)] for val in np.linspace(0, 2*np.pi, 500)], dtype=np.float16)
print(a)

np.savetxt(r'C:\Users\plarotta\software\TSP\data\easy.txt',a)