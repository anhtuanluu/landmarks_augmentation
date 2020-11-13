import json
import cv2

path_annotation = 'data_augment/caotoc_augment/images/1_RECOGNITION_8603_car_30E73245_20200727_134049_f88ed83d35be32d41f1a5e63b2166f80_1.json'
path_image = 'data_augment/caotoc_augment/images/1_RECOGNITION_8603_car_30E73245_20200727_134049_f88ed83d35be32d41f1a5e63b2166f80_1.jpg'

labels = []
with open(path_annotation) as json_file:
    data = json.load(json_file)
    width=data['imageWidth']
    height=data['imageHeight']
    for shape in data['shapes']:
        points=shape["points"]
        (x1,y1),(x2,y2),(x3,y3),(x4,y4)=points

        xc = (x1 + x3) / 2
        yc = (y1 + y3) / 2

        xmin=min(x1,x2,x3,x4)
        ymin=min(y1,y2,y3,y4)
        xmax=max(x1,x2,x3,x4)
        ymax=max(y1,y2,y3,y4)
        
        labels.append([xmin, ymin, xmax, ymax, x1, y1, x2, y2, xc, yc, x3, y3, x4, y4])

image = cv2.imread(path_image, -1)

for label in labels:  
    cv2.circle(image, (int(label[4]), int(label[5])), 1, (0, 0, 255), 4)
    cv2.circle(image, (int(label[6]), int(label[7])), 1, (0, 255, 255), 4)
    cv2.circle(image, (int(label[8]), int(label[9])), 1, (255, 0, 255), 4)
    cv2.circle(image, (int(label[10]), int(label[11])), 1, (0, 255, 0), 4)
    cv2.circle(image, (int(label[12]), int(label[13])), 1, (255, 255, 0), 4)
cv2.imwrite("img.jpg", image)