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

def get_penalty(rob, moves):
    input = np.log(np.array(rob.read_irs()))/10
    sensors = subs_inf(input)
    penalty = 0
    for i in sensors:
        if i < -0.3:
            penalty += 10
        elif i < -0.0:
            penalty += 1
    if moves[0] > 0 and moves[1] > 0:
        penalty -= 2
    if moves[0] < 0 and moves[1] < 0:
        penalty += 4
    return penalty


def cap_moves(move):
    if move <3 and move >= 0:
        move = 3
    elif move > -3 and move < 0:
        move = -3
    return move

def subs_inf(inputs):
    inputs[inputs == -inf] = 1
    return inputs

def simulation(rob, ind, controller, STEPS):
    rob.play_simulation()
    # rob.toggle_visualization()
    rob.wait_for_ping()
    total_fitness = 100
    for i in range(STEPS):
        # print("robobo is at {}".format(rob.position()))
        # print("ROB IRS: {}".format(np.log(np.array(rob.read_irs()))/10))
        inputs = np.log(np.array(rob.read_irs()))/10
        input = subs_inf(inputs[3:])
        if all(i == 1 for i in input):
            print('ALL SAME=', input)
            left, right = randint(1,3), randint(1,3)
            rob.move(left, right, 500)
        else:
            moves = controller.control(input, ind)
            left = cap_moves(moves[0])
            right = cap_moves(moves[1])
            rob.move(left, right, 500)
            penalty= get_penalty(rob, moves)
            total_fitness -= penalty
    
    print('FITNESS=', total_fitness)
    rob.stop_world()
    rob.wait_for_stop()
    # print('SIMULATION STOPPED')
    return total_fitness
