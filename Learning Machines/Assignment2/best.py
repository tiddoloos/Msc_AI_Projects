import numpy as np
from commands_without_green_box import *
from controller import *
import robobo


STEPS=500

f = open('HOF/classification1_fitness=223.0_NGEN=12_steps=300_MU=25_LA=50.txt', 'r')
weights = []
for weight in f:
    weights.append(float(weight))
ind = np.asarray(weights)

n_hidden = 3
n_vars = 4 * 3

rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
controller = Player_controller(n_hidden)

simulation(rob, ind, controller, STEPS)