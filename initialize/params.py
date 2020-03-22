import numpy as np
from scipy.io import loadmat

channel = loadmat('initialize/channel.mat')['H']
channel = np.reshape(channel, (-1,))

h = loadmat('initialize/h.mat')['h']
h = np.reshape(h, (-1,))

bit_allocation = np.hstack((4 * np.ones(255, dtype=np.int), np.zeros(1, dtype=np.int)))

E = loadmat('initialize/subcarrier_energy.mat')['E']
E = np.reshape(E, (-1,))

coef = abs(channel)
coef = coef[0] / coef
coef[-1] = 1
