import torch
import torch.nn as nn
import utils
import cv2 as cv
import gym
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    img = np.array([[[1,2,3]]], dtype=np.int8)
    cv.imshow("img", img)
    cv.waitKey(0)
    print(img.shape)

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    print(gray)