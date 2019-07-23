
#------------------------------------#
# Author: Yueh-Lin Tsou              #
# Update: 7/20/2019                  #
# E-mail: hank630280888@gmail.com    #
#------------------------------------#

"""-----------------------------------------------------------------
Implement Template Matching by using  Normalised Cross Correlation
-------------------------------------------------------------------"""

import argparse
import numpy as np
import cv2
import pylab as pl

# ------------------ Normalised Cross Correlation ------------------ #
def Normalised_Cross_Correlation(roi, target):
    # Normalised Cross Correlation Equation
    cor = np.sum(roi * target)
    nor = np.sqrt( (np.sum(roi ** 2))) * np.sqrt(np.sum(target ** 2))

    return cor / nor

# ----------------------- template matching ----------------------- #
def template_matching(img, target):
    # initial parameter
    height, width = img.shape
    tar_height, tar_width = target.shape
    (max_Y, max_X) = (0, 0)
    MaxValue = 0

    # Set image, target and result value matrix
    img = np.array(img, dtype="int")
    target = np.array(target, dtype="int")
    NccValue = np.zeros((height-tar_height, width-tar_width))

    # calculate value using filter-kind operation from top-left to bottom-right
    for y in range(0, height-tar_height):
        for x in range(0, width-tar_width):
            # image roi
            roi = img[y : y+tar_height, x : x+tar_width]
            # calculate ncc value
            NccValue[y, x] = Normalised_Cross_Correlation(roi, target)
            # find the most match area
            if NccValue[y, x] > MaxValue:
                MaxValue = NccValue[y, x]
                (max_Y, max_X) = (y, x)

    return (max_X, max_Y)


# -------------------------- main -------------------------- #
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to input image")
    ap.add_argument("-t", "--target", required = True, help = "Path to target")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"], 0)
    target = cv2.imread(args["target"], 0)

    height, width = target.shape

    # function
    top_left = template_matching(image, target)
    # draw rectangle on the result region
    cv2.rectangle(image, top_left, (top_left[0] + width, top_left[1] + height), 0, 3)

    # show result
    pl.subplot(111)
    pl.imshow(image)
    pl.title('result')
    pl.show()
