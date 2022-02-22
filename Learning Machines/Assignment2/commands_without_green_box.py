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

def get_reward():
    pass

def move_robot(rob,move):
    if move == 0:
        rob.move(20, 20, 2000)
    elif move == 1:
        rob.move(18, 0, 100)
    else:
        rob.move(0, 18, 100)

def blob(image):
    # detector = cv2.SimpleBlobDetector()
    keypoints = detector.detect(image)
    # print(keypoints)
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

def get_image_input(rob):
    image = rob.get_image_front()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = add_border(image)
    keypoints = blob(image)
    points = cv2.KeyPoint_convert(keypoints)
    # print(points)
    inputs = split_image(points)
    # print(inputs)
    # cv2.imwrite("test_pictures.png",image)
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255)
                                          ,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    return inputs


def simulation(rob, ind, controller, STEPS):
    
    rob.play_simulation()
    rob.wait_for_ping()
    # rob.toggle_visualization()
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

