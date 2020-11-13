import cv2
import glob
import json
import os

def check(x1, x2, x3, x4, y1, y2, y3, y4, h, w):
    if x1 > w or x2 > w or x3 > w or x4 > w or x1 < 150 or x2 < 150 or x3 < 150 or x4 < 150:
        return False
    elif y1 > h or y2 > h or y3 > h or y4 > h or y1 < 50 or y2 < 50 or y3 < 50 or y4 < 50:
        return False
    else:
        return True

path_annotations = glob.glob('/media/tuanlxa/7095b633-ac18-45b1-af43-fa978750a675/datasets/dataplate/landmark/data_plate_split/data_train/caotoc/images/*.json')
save_path = 'caotoc_crop/images/'
# path_annotations = glob.glob('images/*.json')
for path_annotation in path_annotations:
    image_path = path_annotation[:-4] + 'jpg'
    image = cv2.imread(image_path, -1)
    ano_name = os.path.basename(path_annotation)
    image_name = os.path.basename(image_path)

    x = 50
    y = 150

    image_crop = image[x:, y:]
    height, width, c = image_crop.shape
    labels = []

    with open(path_annotation) as json_file:
        data = json.load(json_file)
        for shape in data['shapes']:

            width_ori=data['imageWidth']
            height_ori=data['imageHeight']
            points=shape["points"]

            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = points

            if not check(x1, x2, x3, x4, y1, y2, y3, y4, height_ori, width_ori):
                continue

            xc = (x1 + x3) / 2
            yc = (y1 + y3) / 2

            x1_n = x1 - 150
            x2_n = x2 - 150
            x3_n = x3 - 150
            x4_n = x4 - 150
            y1_n = y1 - 50
            y2_n = y2 - 50
            y3_n = y3 - 50
            y4_n = y4 - 50

            xc = (x1 + x3) / 2
            yc = (y1 + y3) / 2

            xmin=min(x1,x2,x3,x4)
            ymin=min(y1,y2,y3,y4)
            xmax=max(x1,x2,x3,x4)
            ymax=max(y1,y2,y3,y4)

            labels.append([xmin, ymin, xmax, ymax, x1_n, y1_n, x2_n, y2_n, xc, yc, x3_n, y3_n,x4_n, y4_n])
    data = {
                "version": "4.5.6",
            }
    data['flags'] = {}
    data['shapes'] = []
    data['imageWidth'] = width
    data['imageHeight'] = height
    data['imageData'] = None
    data['imagePath'] =  image_name
            
    for label in labels:
        data_point = [[label[4],label[5]],
                        [label[6],label[7]],
                        [label[10],label[11]],
                        [label[12],label[13]]]
        print(data_point)
        data['shapes'].append({'label': 'plate', 
                            'points': data_point, 
                            'group_id': None,
                            'shape_type': 'polygon',
                            'flags': {}})

        # cv2.circle(image_crop, (int(label[4]), int(label[5])), 1, (0, 0, 255), 4)
        # cv2.circle(image_crop, (int(label[6]), int(label[7])), 1, (0, 255, 255), 4)
        # cv2.circle(image_crop, (int(label[10]), int(label[11])), 1, (0, 255, 0), 4)
        # cv2.circle(image_crop, (int(label[12]), int(label[13])), 1, (255, 255, 0), 4)
    if(len(data['shapes']) == 0):
        continue
    cv2.imwrite(save_path + image_name, image_crop)
    with open(save_path + ano_name, 'w') as outfile:
        json.dump(data, outfile, indent=4, separators=(",", ": "), sort_keys=True)
    # break

