import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras, sys, time, os, warnings 
import numpy as np
import pandas as pd
import cv2
import json
from keras.preprocessing.image import ImageDataGenerator,  img_to_array, load_img
from function_backup import ImageDataGenerator_landmarks
import os 
from PIL import Image
import base64

path_annotation = 'images/1_RECOGNITION_19796_car_30E28204_20200728_024212_d5bc78169ce00cab2c541c901d8c6f96.json'
path_image = 'images/1_RECOGNITION_19796_car_30E28204_20200728_024212_d5bc78169ce00cab2c541c901d8c6f96.jpg'
base_name = os.path.basename(path_image)
save_dir = 'test_image/'
landmarks = ["1","2","0","3","4"]
labels = []

with open(path_annotation) as json_file:
    
    data = json.load(json_file)
    width=data['imageWidth']
    height=data['imageHeight']
    shapes = data['shapes']
    imagePaths = data['imagePath']

    for shape in shapes:
        points=shape["points"]
        (x1,y1),(x2,y2),(x3,y3),(x4,y4)=points

        xc = (x1 + x3) / 2
        yc = (y1 + y3) / 2

        xmin=min(x1,x2,x3,x4)
        ymin=min(y1,y2,y3,y4)
        xmax=max(x1,x2,x3,x4)
        ymax=max(y1,y2,y3,y4)
        
        # labels.append([(int(xmin), int(ymin), int(xmax), int(ymax)), 
        #                 (int(x1), int(y1)), 
        #                 (int(x2), int(y2)), 
        #                 (int(xc), int(yc)), 
        #                 (int(x3), int(y3)), 
        #                 (int(x4), int(y4))])
        labels.append([(int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (int(xc), int(yc)), 
                        (int(x3), int(y3)), 
                        (int(x4), int(y4))])

img_bd = cv2.imread(path_image, -1)
# byte_image = Image.open(path_image)
# byte_image.tobytes().decode("latin1")
# with open(path_image, "rb") as f:
#     data = f.read()
#     img_file_data = base64.b64encode(data)
# print(img_file_data)

landmark_bd = labels[0]
# landmark_bd = landmark_bd[1:]

datagen = ImageDataGenerator(#rotation_range=15,
                             # width_shift_range=10, 
                             # height_shift_range=10.0,  
                             ## Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
                             # shear_range=5.0,
                             ## zoom_range: Float or [lower, upper]. 
                             ## Range for random zoom. If a float, 
                             ## [lower, upper] = [1-zoom_range, 1+zoom_range]
                             zoom_range=[0.6, 1.1], 
                             fill_mode='nearest', 
                             # cval=-2, 
                             horizontal_flip=False, 
                             vertical_flip=False)

generator = ImageDataGenerator_landmarks(datagen,
                                         ignore_horizontal_flip=True,
                                         target_shape=None,
                                         loc_xRE=0, 
                                         loc_xLE=2,
                                         flip_indicies =  ((0,2),
                                                          (1,3),
                                                          (6,8),
                                                          (7,9))
                                        )
xy = np.array([generator.get_ymask(img_bd,landmark_bd)])
# print('landmarks: {}'.format(landmark_bd))

# plt.figure(figsize=(6,6))
# plt.imshow(xy[0][:,:,3])
# plt.show()
# plt.close('all')

def singleplot(ax,x,y):
    ax.imshow(x/255.0)

    colors = ['b','g','r','c','m']
    for i, marker,c in zip(range(0,len(y),2),
                           landmarks,
                           colors):
        # print((y[i],y[i+1]))      
        ax.annotate(marker,
                    (y[i],y[i+1]),
                    color=c)
        
def pannelplot(figID=0, dir_image=None, nrow_plot = 2,ncol_plot = 2, fignm="fig",save=True):        
    
    fig = plt.figure(figsize=(15,15))
    #fig.subplots_adjust(hspace=0,wspace=0)
    xs, ys = [], [] 
    count = 1
    for x_train,y_train in generator.flow(xy,batch_size=1):
        
        i = 0
        if len(x_train) == 1:
            ax = fig.add_subplot(nrow_plot,ncol_plot,count)
            #ax.axis("off")
            singleplot(ax, x_train[0], y_train[0])
            
            # save image & json
            data_point = [[y_train[0][0],y_train[0][1]],
                            [y_train[0][2],y_train[0][3]],
                            [y_train[0][6],y_train[0][7]],
                            [y_train[0][8],y_train[0][9]]]
            
            cv2.imwrite(save_dir + base_name[:-4] + '_' + str(count) + '.jpg', x_train[0])
            data = {
                "version": "4.5.6",
            }
            data['flags'] = {}
            data['shapes'] = []
            data['shapes'].append({'label': 'plate', 
                                    'points': data_point, 
                                    'group_id': None,
                                    'shape_type': 'polygon',
                                    'flags': {}})
            data['imageWidth'] = width
            data['imageHeight'] = height
            data['imageData'] = None
            data['imagePath'] = base_name[:-4] + '_' + str(count) + '.jpg'
            with open(save_dir + base_name[:-4] + '_' + str(count) + '.json', 'w') as outfile:
                json.dump(data, outfile, indent=4, separators=(",", ": "), sort_keys=True)
            
            # print(data)
        
            if count == nrow_plot * ncol_plot:
                break
            count += 1
    if save:
         plt.savefig(dir_image + "/fig{:04.0f}.png".format(figID),
                     bbox_inches='tight',pad_inches=0)   
    else:
        plt.show()

pannelplot(save=False)