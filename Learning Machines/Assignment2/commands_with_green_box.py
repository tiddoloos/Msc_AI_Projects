from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
from controller import *

detector = cv2.SimpleBlobDetector_create()

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def move_robot(rob,move):
    if move == 0:
        rob.move(20, 20, 200)
    elif move == 1:
        rob.move(18, 5, 100)
    else:
        rob.move(5, 18, 100)

def blob(image):
    keypoints = detector.detect(image)
    return keypoints

def split_image(points):
    inputs = np.zeros(3)
    if points != ():
        for point in points:
            if point[0] <= 56:
                inputs[0] = 1
            elif point[0] > 56 and point[0] < 112:
                inputs[1] = 1
            elif point[0] >= 112:
                inputs[2] = 1
    return inputs

def add_border(image):
    row, col = image.shape[:2]
    bottom = image[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]
    border_image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
    return border_image

def check_close_blob(image):
    green_count = 0
    h, w, c = image.shape
    h = int(h)
    w = int(round(w/2))
    for i in range(w - 2, w + 2, 1):
        for j in range(h - 4, h, 1):
            b_g = int(image[i][j][1]) - int(image[i][j][0])
            r_g = int(image[i][j][1]) - int(image[i][j][2])
            if b_g > 100 and r_g > 100:
                green_count += 1
    if green_count == 16:
        return 1
    else:
        return 0
    
def get_image_input(rob):
    image = rob.get_image_front()
    close_blob = check_close_blob(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = add_border(image)
    keypoints = blob(image)
    points = cv2.KeyPoint_convert(keypoints)
    inputs = split_image(points)
    inputs = np.append(inputs, close_blob)
    return inputs

def simulation(rob, ind, controller, STEPS):
    rob.play_simulation()
    rob.wait_for_ping()
    rob.toggle_visualization()
    rob.set_phone_tilt(25.8, 10)
    fitness = 0
    for i in range(STEPS):
        inputs = get_image_input(rob)
        move = controller.control(inputs, ind)
        move_robot(rob, move)
        if rob.collected_food() == 7:
            fitness += 1   
    fitness += rob.collected_food()
    print('FITNESS=', fitness)
    rob.stop_world()
    rob.wait_for_stop()
    
    return fitness

