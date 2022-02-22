#!/usr/bin/env python3
from __future__ import print_function
from doctest import set_unittest_reportflags
from turtle import pen
import numpy as np
from numpy import inf
from controller_classification import *
from random import *
import cv2
import sys

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def get_penalty(move, sensors):
    penalty = 0
    for i in sensors:
        if i == 1:
            penalty += 7
    if move == 0:
        penalty -= 10
    return penalty


def move_robot(rob,move):
    if move == 0:
        rob.move(10, 10, 100)
    elif move == 1:
        rob.move(18, 0, 100)
    else:
        rob.move(0, 18, 100)

def get_sensors(rob):
    inputs = np.log(np.array(rob.read_irs()))/10
    inputs = np.array([inputs[3], inputs[5], inputs[7]])
    inputs[inputs == -inf] = 0
    inputs[inputs > -0.1] = 0
    inputs[inputs <= -0.1] = 1
    return inputs

def simulation(rob, ind, controller, STEPS):
    rob.play_simulation()
    # rob.toggle_visualization()
    rob.wait_for_ping()
    # rand_time= randint(0,2500)
    # rob.move(20, -20, rand_time)
    print("RANDOM SPIN DONE")
    total_fitness = 0
    for i in range(STEPS):
        sensors = get_sensors(rob)
        move = controller.control(sensors, ind)
        move_robot(rob, move)
        sensors = get_sensors(rob)
        penalty= get_penalty(move, sensors)
        total_fitness -= penalty
    
    # print('FITNESS=', total_fitness)
    rob.stop_world()
    rob.wait_for_stop()
    # print('SIMULATION STOPPED')
    return total_fitness
