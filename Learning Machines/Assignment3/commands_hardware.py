from __future__ import print_function
from re import search
import numpy as np
import robobo
import cv2
import sys
import signal
from controller_hidden import *
import os

params = cv2.SimpleBlobDetector_Params()
params.minArea = 100
params.maxArea = 2200000
# params.filterByConvexity = False
params.filterByInertia = False
# params.filterByCircularity = Tr
detector = cv2.SimpleBlobDetector_create(params)


class Image():
    def __init__(self, image):
        self.image = image
        self.hsv = None
        self.masked_image = None
        self.green_mask = None
        self.red_mask = None
        self.keypoints = None
        self.inputs = None

    def initialize(self):
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.green_mask = cv2.inRange(self.hsv, (35, 25, 25), (80, 255, 255))
        mask1 = cv2.inRange(self.hsv, (0, 50, 50), (10, 255, 255))
        mask2 = cv2.inRange(self.hsv, (170, 50, 50), (180, 255, 255))
        self.red_mask = mask1 + mask2

    def mask_image(self):
        mask = self.green_mask + self.red_mask
        imask = mask > 0
        self.masked_image = np.zeros_like(self.hsv, np.uint8)
        self.masked_image[imask] = self.image[imask]
        self.masked_image[np.where((self.masked_image == [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    def blob(self):
        points = detector.detect(self.masked_image)
        self.keypoints = cv2.KeyPoint_convert(points)
        im_with_keypoints = cv2.drawKeypoints(self.masked_image, points, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite('TEST_LAB1.jpg', im_with_keypoints)
        # cv2.waitKey(0)

    def derive_inputs(self):
        inputs = self.search_pixels()
        block = self.has_block()
        inputs = np.append(inputs, block)
        self.inputs = inputs

    def search_pixels(self):
        inputs = np.zeros(6, dtype=int)
        if self.keypoints != ():
            for point in self.keypoints:

                w, h = point[0], point[1]
                w = int(w)
                h = int(h)
                count_red = 0
                count_green = 0
                for i in range(h - 1, h + 2, 1):
                    for j in range(w - 1, w + 2, 1):
                        if self.red_mask[i][j] > 0:
                            count_red += 1

                        elif self.green_mask[i][j] > 0:
                            count_green += 1

                if count_red >= 4:
                    index_red = self.split_image(w)
                    inputs[index_red] = int(1)
                elif count_green >= 4:
                    index_green = self.split_image(w)
                    inputs[index_green + 3] = int(1)
        return inputs

    def split_image(self, point):
        if point <= 174:
            index = 0
        elif point > 174 and point < 347:
            index = 1
        elif point >= 347:
            index = 2
        return index

    def has_block(self):
        h, w, c = self.image.shape
        print(self.image.shape)
        count = 0
        h = int(h)
        w = int(round(w / 2))
        for i in range(250, 260, 1):
            for j in range(w - 5, w + 5, 1):
                if self.red_mask[i][j] > 0:
                    count += 1
                self.image[i][j] = [0, 0, 0]
        # print(count)
        # cv2.imwrite("TEST2.png", self.image)

        if count >= 50:
            print("BLOCK GRABBED")
            return 1
        else:
            return 0

    def add_border(self):
        self.image = cv2.copyMakeBorder(self.image, 0, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        self.masked_image = cv2.copyMakeBorder(self.masked_image, 0, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def move_robot(rob, move):
    if move == 0:
        rob.move(20, 20, 600)
    elif move == 1:
        rob.move(18, 5, 400)
    else:
        rob.move(5, 18, 400)


def get_image_input(rob):
    im = rob.get_image_front()
    im = im[:560,0:480]
    cv2.imwrite('cut.png', im)
    image = Image(im)
    image.add_border()
    image.initialize()
    image.mask_image()
    image.blob()
    image.derive_inputs()
    print(image.inputs)
    return image.inputs


def simulation(rob, ind, controller, STEPS):
    signal.signal(signal.SIGINT, terminate_program)

    rob.set_phone_tilt(120, 10)
    # inputs = get_image_input(rob)
    # print(inputs)
    for i in range(STEPS):
        inputs = get_image_input(rob)
        move = controller.control(inputs, ind)
        move_robot(rob, move)




# simulation()