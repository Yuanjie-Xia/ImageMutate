import statistics

import matplotlib.image as img
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image
import numpy as np
import os
import time
import random
from sklearn.linear_model import LinearRegression
file1 = open('print.txt', 'r')
count = 0
x1 = []
x2 = []
y = []

while True:
    count += 1
    # Get next line from file
    line = file1.readline()
    # if line is empty
    # end of file is reached
    if not line:
        break
    if count>1:
        item = line.split()
        file_size = item[0]
        file_name = item[1]
        image = img.imread('ILSVRC/Data/DET/test/' + str(file_name))
        if len(image.shape) == 3:
            leng = image.shape[0]
            wid = image.shape[1]
            if leng>64:
                if wid>64:
                    r_length = random.randint(0, leng - 64)
                    r_width = random.randint(0, wid - 64)
                    cut_image = image[r_length:r_length + 64, r_width:r_width + 64, :]
                    # print(cut_image.shape)
                    img0 = Image.fromarray(cut_image)
                    img0.save('split_image0.JPEG')
                    # if file_size == str(108):
                    # print(file_size)
                    # if leng * wid == 250000:
                    os.system('rm -rf 001.bmp')
                    start_time = time.time()
                    os.system(
                        'djpeg -colors 256 -scale 1/4 -bmp -dct float split_image0.JPEG>001.bmp')
                    running_time = time.time() - start_time
                    # print(running_time)
                    # my_dict[file_name] = (int(file_size), running_time)
                    x1.append(int(file_size))
                    x2.append(leng * wid)
                    y.append(running_time * 1000)

#print(x2)
print(statistics.stdev(y)/statistics.mean(y))
plt.plot(x1, y, 'o')
plt.show()
# plt.plot(x2, y, 'o')
# plt.show()


