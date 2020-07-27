import torch
import torch.nn as nn
import utils
import cv2 as cv
import gym
import numpy as np


if __name__ == "__main__":

    task = "Breakout-v0"
    env = gym.make(task)
    o = env.reset()

    ori_size = (110,84)
    gray = cv.cvtColor(o, cv.COLOR_RGB2GRAY)
    gray = downsampling(gray, ori_size)
    print(gray.shape)

    crop_size = (84,84)
    [h_d, w_d] = (np.subtract(ori_size, crop_size) / 2).astype(np.int)
    cropped = gray[h_d + 5:ori_size[0]-h_d+5, w_d:ori_size[1]-w_d]
    print(cropped.shape)

    cv.imshow("color", o)
    cv.waitKey(0)
    cv.imshow("gray", gray)
    cv.waitKey(0)
    cv.imshow("crop", cropped)
    cv.waitKey(0)