#!/usr/bin/env python3
from __future__ import print_function
from doctest import set_unittest_reportflags
from turtle import pen
import numpy as np
from numpy import inf
from controller_classification_hardware import *
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
        penalty -= 5
    return penalty


def move_robot(rob,move):
    if move == 0:
        rob.move(10, 10, 600)
    elif move == 1:
        rob.move(18, -2, 600)
    else:
        rob.move(-2, 18, 600)

def get_sensors(rob):
    inputs = np.log(np.array(rob.read_irs()))/10
    inputs = np.array([inputs[3], inputs[5], inputs[7]])
    inputs = inputs * -1
    inputs[inputs == -inf] = 0
    inputs[inputs > -0.3] = 0  # stond 0.1
    inputs[inputs <= -0.3] = 1  # stond 0.1
    return inputs

def simulation(rob, ind, controller, STEPS):

    for i in range(STEPS):
        sensors = get_sensors(rob)
        move = controller.control(sensors, ind)

        move_robot(rob, move)

