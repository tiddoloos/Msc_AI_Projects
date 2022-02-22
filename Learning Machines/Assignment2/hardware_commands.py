from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey
from controller import *


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 70
params.maxThreshold = 120
params.minArea = 200
params.maxArea = 100000000
detector = cv2.SimpleBlobDetector_create(params)

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

def get_reward():
    pass

def move_robot(rob,move):
    if move == 0:
        rob.move(20, 20, 3000)
    elif move == 1:
        rob.move(18, 4, 600)
    else:
        rob.move(4, 18, 600)

def blob(image):
    # detector = cv2.SimpleBlobDetector()
    keypoints = detector.detect(image)
    # print(keypoints)
    return keypoints

def split_image(points):
    inputs = np.zeros(3)
    
    if points != ():
        for point in points:
            print(point)
        if point[0] <= 160:
            inputs[0] = 1
        elif point[0] > 160 and point[0] < 320:
            inputs[1] = 1
        elif point[0] >= 320:
            inputs[2] = 1
    return inputs

def add_border(image):
    row, col = image.shape[:2]
    bottom = image[row-2:row, 0:col]
    border_image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
    return border_image

def check_close_blob(image):
    green_count = 0
    # print(image.shape)
    w, h, c = image.shape
    h = int(h)
    w = int(round(w/2))
    pixel_list = []
    for i in range(w - 2, w + 2, 1):
        for j in range(h - 4, h, 1):
            pixel_list.append(image[i][j][1])
    if (sum(pixel_list)/len(pixel_list)) >= 180:
        print('GREEN')
        return 1
    else:
        return 0
    #         b_g = int(image[i][j][1]) - int(image[i][j][0])
    #         r_g = int(image[i][j][1]) - int(image[i][j][2])
    #         if b_g > 100 and r_g > 100:
    #             print(image[i][j])
    #             green_count += 1
    # if green_count == 16:
    #     print("ALL GREEN")
    #     return 1
    # else:
    #     return 0

def make_green(image):
    w, h, c = image.shape
    h = int(h)
    w = int(round(w/2))
    for i in range(w):
        for j in range(h):
            image[i][j][0]=0
            image[i][j][2]=0
    return image

def get_image_input(rob):
    image = rob.get_image_front()
    crop_image = image[100:, 0:480]
    blurred_image = cv2.GaussianBlur(image, (99, 99), 0)
    # print(crop_image.shape)
    close_blob = check_close_blob(blurred_image)
    image = make_green(crop_image)
    image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    image = add_border(image)
    image = cv2.GaussianBlur(image, (99, 99), 0)
    keypoints = blob(image)
    points = cv2.KeyPoint_convert(keypoints)
    inputs = split_image(points)
    inputs = np.append(inputs, close_blob)


    # # cv2.imwrite("test_image1.png",image)
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255)
                                          ,cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite("ARENA_TEST.png", im_with_keypoints)
    return inputs


def simulation(rob, ind, controller, STEPS):
    signal.signal(signal.SIGINT, terminate_program)
    # rob.play_simulation()
    # rob.wait_for_ping()

    rob.set_phone_tilt(120, 10)
    # rob.pause(6)
    for i in range(STEPS):
        inputs = get_image_input(rob)
        # print(inputs)
        move = controller.control(inputs, ind)
        move_robot(rob, move)
    # fitness = rob.collected_food()
    # print('FITNESS=', fitness)
    # rob.stop_world()
    # rob.wait_for_stop()
    
    # return fitness

