import os
import subprocess
import time
import statistics

import scipy as scipy

os.system('ls -s > print.txt')

file1 = open('print.txt', 'r')
count = 0
list = []
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
        if file_name != '001.JPEG':
            if file_size == str(8):
                os.system('rm -rf 001.JPEG')
                start_time = time.time()
                os.system('djpeg -colors 256 -scale 1/4 '+ str(file_name) +'>001.JPEG')
                running_time = time.time() - start_time
                print(running_time)
                list.append(running_time)
for idx, element in enumerate(list):
    list[idx] = (list[idx]-min(list))/(max(list)-min(list))
print(list)
print(statistics.stdev(list))
file1.close()

