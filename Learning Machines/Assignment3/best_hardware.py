import numpy as np
from commands_hardware import *
from controller_hidden import *
import robobo


STEPS=500

f = open('./src/HOF/classification1_fitness=1.7445754499330417_NGEN=6_steps=100_MU=20_LA=40.txt', 'r')
weights = []
for weight in f:
    weights.append(float(weight))
ind = np.asarray(weights)
# ind =0
n_hidden = 4
n_vars = 4 * 3

rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.246")
controller = Player_controller(n_hidden)

simulation(rob, ind, controller, STEPS)