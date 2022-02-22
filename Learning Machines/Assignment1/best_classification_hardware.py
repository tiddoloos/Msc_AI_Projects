import numpy as np
from commands_classification_hardware import *
from controller_classification_hardware import *
import robobo


STEPS=1000

f = open('./src/HOF/classification_best.txt', 'r')
weights = []
for weight in f:
    weights.append(float(weight))
ind = np.asarray(weights)

n_hidden = 3
n_vars = (4 * n_hidden)

# rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
rob = robobo.HardwareRobobo().connect(address='192.168.43.28')
controller = Player_controller(n_hidden)
simulation(rob, ind, controller, STEPS)