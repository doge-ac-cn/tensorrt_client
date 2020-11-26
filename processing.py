from boundingbox import BoundingBox

import cv2
import numpy as np

INPUT_HEIGHT = 640
INPUT_WIDTH = 640

def preprocess_suit_img(image,input_height,input_width):
    image = cv2.resize(image, ( input_width,input_height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(np.array(image, dtype=np.float32, order='C'), (2, 0, 1))
    image /= 255.0
    return image
def  preprocess(img, new_shape=(640, 640), color=(128, 128, 128), auto=True, scaleFill=False, scaleup=True) :
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # if auto:  # minimum rectangle
    #     dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    # elif scaleFill:  # stretch
    #     dw, dh = 0.0, 0.0
    #     new_unpad = (new_shape[1], new_shape[0])
    #     ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = np.array(img, dtype=np.float32)


    return img/255.0,dw,dh,dw/new_shape[1],dh/new_shape[0]

def nms(boxes, box_confidences, nms_threshold=0.5):
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            iou = intersection / union

            indexes = np.where(iou <= nms_threshold)[0]
            ordered = ordered[indexes + 1]
        keep = np.array(keep).astype(int)
        return keep

def postprocess(buffer, image_width, image_height,dw,dh,padding_w,padding_h,conf_threshold=0.8, nms_threshold=0.5):
    detected_objects = []
    img_scale = [image_width / (INPUT_WIDTH-dw*2), image_height / (INPUT_HEIGHT -dh*2), image_width / (INPUT_WIDTH-dw*2), image_height / (INPUT_HEIGHT-dh*2)]
    num_bboxes = int(buffer[0, 0, 0, 0])

    if num_bboxes:
        bboxes = buffer[0, 1 : (num_bboxes * 6 + 1), 0, 0].reshape(-1, 6)
        # print(bboxes)
        labels = set(bboxes[:, 5].astype(int))
        for label in labels:
            selected_bboxes = bboxes[np.where((bboxes[:, 5] == label) & ((bboxes[:, 4] ) >= conf_threshold))]
            selected_bboxes_keep = selected_bboxes[nms(selected_bboxes[:, :4], selected_bboxes[:, 4] , nms_threshold)]
            for idx in range(selected_bboxes_keep.shape[0]):
                print (selected_bboxes_keep[idx, :2])
                box_xy = ( (selected_bboxes_keep[idx, :2]) \
                    -[dw,dh] ) 
                print(box_xy)
                box_wh = selected_bboxes_keep[idx, 2:4]
                score = selected_bboxes_keep[idx, 4] 
                # print(score)
                box_x1y1 = np.maximum([0,0],box_xy - (box_wh / 2))
                box_x2y2 = np.minimum(box_xy + (box_wh / 2), [INPUT_WIDTH, INPUT_HEIGHT]) 
                box = np.concatenate([box_x1y1, box_x2y2])
                box *= img_scale
                if box[0] == box[2]:
                    continue
                if box[1] == box[3]:
                    continue

                detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], image_height, image_width))
    return detected_objects