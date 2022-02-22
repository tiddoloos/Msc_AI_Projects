import numpy as np
from commands import *
from controller import *
import robobo

STEPS=1000

f = open('HOF/fitness=124.0_NGEN=10_steps=500_MU=10_LA=20.txt', 'r')
weights = []
for weight in f:
    weights.append(float(weight))
ind = np.asarray(weights)

n_hidden = 10
n_vars = (9 * n_hidden) + ((n_hidden + 1)*2) 

rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
controller = Player_controller(n_hidden)

simulation(rob, ind, controller, STEPS)