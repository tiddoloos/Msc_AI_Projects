#!/usr/bin/env python3
from __future__ import print_function
from random import random
from turtle import pen
import numpy as np
from numpy import inf
from controller import *
from random import *
import cv2
import sys

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def cap_moves(move):
    if move <3 and move >= 0:
        move = 3
    elif move > -3 and move < 0:
        move = -3
    return move

def process_sensors(inputs):
    inputs[inputs == -inf] = -1
    inputs = inputs * - 1
    inputs = np.array([0 if x > -0.4 else x for x in inputs])
    return inputs

def simulation(rob, ind, controller, STEPS):
    for i in range(STEPS):
        inputs = np.log(np.array(rob.read_irs()))/10
        input = process_sensors(inputs[3:])
        if all(i == 1 for i in input):
            print('ALL SAME=', input)
            left, right = randint(1,3), randint(1,3)
            rob.move(left, right, 500)
        else:
            moves = controller.control(input, ind)
            left = cap_moves(moves[0])
            right = cap_moves(moves[1])
            rob.move(left, right, 500)
    rob.stop_world()
    rob.wait_for_stop()
    # print('SIMULATION STOPPED')
