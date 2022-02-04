import cv2
import numpy as np

file1 = open('print.txt', 'r')
count = 0
max_col = 0
max_wid = 0
while True:
    count += 1
    # Get next line from file
    line = file1.readline()
    # if line is empty
    # end of file is reached
    if not line:
        break
    if count > 1:
        item = line.split()
        image_address = item[1]
        img = cv2.imread("ILSVRC/Data/DET/test/"+image_address)
        #print(img)
        col, wid, rgb = img.shape
        img = cv2.copyMakeBorder(img, 0, 500-col, 0, 500-wid, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        col, wid, rgb = img.shape
        print(col)
        print(wid)
        max_col = max(max_col,col)
        max_wid = max(max_wid,wid)

print(max_col)
print(max_wid)


#ori0 = cv2.imread("ILSVRC/Data/DET/test/ILSVRC2017_test_00000001.JPEG")
#ori0 = cv2.cvtColor(ori0, cv2.COLOR_BGR2RGB)
#ori0 = np.asarray(ori0)
#print(ori0.shape)
#print(ori0)
#ori0 = ori0.transpose(2,0,1)
#print(ori0.shape)
#print(ori0)
#ori0 = ori0.reshape(3,-1)
#print(ori0.shape)
#print(ori0)
#ori0 = ori0.reshape(1, -1)
#print(ori0.shape)
#print(ori0)
configuration_set = [['-bmp', '-gif', 'os2', '-pnm'],['-scale 1/2', '-scale 1/4', '-scale 1/8'],
                         ['-dct int', '-dct fast', '-dct float'],
                         ['-dither fs', '-dither ordered', '-dither none'],
                         ['-nosmooth']]
print(len(configuration_set))