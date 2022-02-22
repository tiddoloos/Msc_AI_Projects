import numpy as np
from commands_classification import *
from controller_classification import *
import robobo


STEPS=400

f = open('HOF/classification_best.txt', 'r')
weights = []
for weight in f:
    weights.append(float(weight))
ind = np.asarray(weights)

n_hidden = 3
n_vars = 4 * 3

rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
controller = Player_controller(n_hidden)

simulation(rob, ind, controller, STEPS)