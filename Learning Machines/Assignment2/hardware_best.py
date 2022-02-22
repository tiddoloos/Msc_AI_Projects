import numpy as np
from hardware_commands import *
from hardware_controller import *
import robobo


STEPS=400

f = open('./src/HOF/BEST_268.txt', 'r')
weights = []
for weight in f:
    weights.append(float(weight))
ind = np.asarray(weights)

n_hidden = 3
n_vars = 5 * 3

# rob = robobo.SimulationRobobo().connect(address='127.0.0.1', port=19997)
rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.246")
controller = Player_controller(n_hidden)

simulation(rob, ind, controller, STEPS)