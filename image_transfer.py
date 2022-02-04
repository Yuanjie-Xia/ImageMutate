import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import csv
from random import randint as ri
import math


def rotate(address):
    # read the input image
    img = cv2.imread(address)
    # convert from BGR to RGB so we can plot using matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # disable x & y axis
    # plt.axis('off')
    # show the image
    # plt.imshow(img)
    # plt.show()
    # get the image shape
    rows, cols, dim = img.shape
    # angle from degree to radian
    angle = np.radians(10)
    # transformation matrix for Rotation
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    # apply a perspective transformation to the image
    rotated_img = cv2.warpPerspective(img, M, (int(cols), int(rows)))
    # compare(img, rotated_img, address)


def shearing(address):
    img = cv2.imread(address)
    # convert from BGR to RGB so we can plot using matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # get the image shape
    rows, cols, dim = img.shape
    # transformation matrix for Shearing
    # shearing applied to x-axis
    M = np.float32([[1, 0.5, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    # shearing applied to y-axis
    # M = np.float32([[1,   0, 0],
    #             	  [0.5, 1, 0],
    #             	  [0,   0, 1]])
    # apply a perspective transformation to the image
    sheared_img = cv2.warpPerspective(img, M, (int(cols * 1.5), int(rows * 1.5)))
    # compare(img, sheared_img, address)


def scaling(address):
    img = cv2.imread(address)
    # convert from BGR to RGB so we can plot using matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # get the image shape
    rows, cols, dim = img.shape
    # transformation matrix for Scaling
    M = np.float32([[1.5, 0, 0],
                    [0, 1.8, 0],
                    [0, 0, 1]])
    # apply a perspective transformation to the image
    scaled_img = cv2.warpPerspective(img, M, (cols * 2, rows * 2))
    # save the resulting image to disk
    # compare(img, scaled_img, address)


def all_position_convert(img):
    # convert from BGR to RGB so we can plot using matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols, dim = img.shape
    # img = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # get the image shape
    # transformation matrix for Scaling
    points1 = np.float32([[0, 0], [0, rows], [cols, 0], [rows, cols]])
    rand1 = ri(0, rows)
    rand2 = ri(0, cols)
    rand3 = ri(0, rows)
    rand4 = ri(0, cols)
    rand5 = ri(0, rows)
    rand6 = ri(0, cols)
    rand7 = ri(0, rows)
    rand8 = ri(0, cols)
    points2 = np.float32([[rand1, rand2], [rand3, rand4], [rand5, rand6], [rand7, rand8]])
    # applying getPerspectiveTransform() function to transform the perspective of the given source image to the corresponding points in the destination image
    resultimage = cv2.getPerspectiveTransform(points1, points2)
    # applying warpPerspective() function to fit the size of the resulting image from getPerspectiveTransform() function to the size of source image
    finalimage = cv2.warpPerspective(img, resultimage, (rows, cols))
    # plt.imshow(finalimage)
    # plt.show()
    return img, finalimage

    # save the resulting image to disk
    # compare(img, scaled_img, address)


def smooth(img):
    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)
    # save the resulting image to disk
    # compare(img, dst, address)
    return img, dst


def gaussianFiltering(img):
    dst = cv2.GaussianBlur(img, (5, 5), 0)
    # save the resulting image to disk
    # compare(img, dst, address)
    return img, dst


def medianblur(img):
    dst = cv2.medianBlur(img, 5)
    # save the resulting image to disk
    # compare(img, dst, address)
    return img, dst


def bilateralFiltering(img):
    dst = cv2.cv2.bilateralFilter(img, 9, 75, 75)
    # save the resulting image to disk
    # compare(img, dst, address)
    return img, dst


def reverse(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = img
    rows, cols, dim = img.shape
    for i in range(rows):
        for j in range(cols):
            if i < j:
                for k in range(dim):
                    # print(dst[i, j, k])
                    dst[i, j, k] = ri(0,255)
    # plt.imshow(dst)
    # plt.show()
    # compare(img, dst, address)
    return img, dst


def compare(original_img, tran_img, address):
    plt.imsave("original_image.JPEG", original_img)
    plt.imsave("transformed_image.JPEG", tran_img)
    print('transformed image:')
    start_time = time.time()
    os.system('djpeg -colors 256 -bmp -scale 1/4 transformed_image.JPEG>002.bmp')
    running_time1 = time.time() - start_time
    print(running_time1)
    # time.sleep(0.5)
    print('origianl image:')
    start_time = time.time()
    os.system('djpeg -colors 256 -bmp -scale 1/4 original_image.JPEG>001.bmp')
    running_time2 = time.time() - start_time
    print(running_time2)
    with open("djpeg_reverse_performance.csv", 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([address, str(running_time2), str(running_time1)])


# disable x & y axis
# plt.axis('off')
# show the resulting image
# plt.imshow(rotated_img)
# plt.show()
# save the resulting image to disk
# plt.imsave("city_rotated.jpg", rotated_img)



