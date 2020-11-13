import numpy as np
import json
import cv2
import glob

ximg = np.ones((6,6,3))

yimg = np.zeros((6,6,2))
yimg[:] = -1
yimg[3,3] = 0, 3
yimg[2,2] = 0, 3
yimg[1,1] = 1, 3
image = np.dstack([ximg,yimg])
image = np.expand_dims(image, axis=0)
print(image)
for irow in range(image.shape[0]):
    yobj = image[irow,:,:,4]
    ymask = image[irow,:,:,3]
print(ymask)

ix, iy = np.where(yobj==3)
print(ix, iy)
for i, j in zip(ix, iy):
    if ymask[i,j]==0:
        print("true")