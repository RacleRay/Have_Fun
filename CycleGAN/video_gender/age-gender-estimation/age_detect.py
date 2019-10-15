# -*- coding: utf-8 -*-

from wide_resnet import WideResNet
import numpy as np
import cv2
import dlib
import pickle

depth = 16
width = 8
img_size = 64
model = WideResNet(img_size, depth=depth, k=width)()
model.load_weights('weights.hdf5')

detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture('../data/video.mp4')

pos = []
frame_id = -1

while cap.isOpened():
    ret, image_np = cap.read()
    frame_id += 1
    if len((np.array(image_np)).shape) == 0:
        break

    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    img_h = image_np.shape[0]
    img_w = image_np.shape[1]
    detected = detector(image_np, 1)

    if len(detected) > 0:
        for d in detected:
            x0, y0, x1, y1, w, h = d.left(), d.top(), d.right(), d.bottom(), d.width(), d.height()
            x0 = max(int(x0 - 0.25 * w), 0)
            y0 = max(int(y0 - 0.45 * h), 0)
            x1 = min(int(x1 + 0.25 * w), img_w - 1)
            y1 = min(int(y1 + 0.05 * h), img_h - 1)
            w = x1 - x0
            h = y1 - y0
            if w > h:
                x0 = x0 + w // 2 - h // 2
                w = h
                x1 = x0 + w
            else:
                y0 = y0 + h // 2 - w // 2
                h = w
                y1 = y0 + h
            
            face = cv2.resize(image_np[y0: y1, x0: x1, :], (img_size, img_size))
            result = model.predict(np.array([face]))
            # 只取性别
            pred_gender = result[0][0][0]

            if pred_gender > 0.5:
                # frame_id，位置，大小，性别
                pos.append([frame_id, y0, y1, x0, x1, h, w, 'F'])
            else:
                pos.append([frame_id, y0, y1, x0, x1, h, w, 'M'])

print(frame_id + 1, len(pos))

with open('../data/pos.pkl', 'wb') as fw:
    pickle.dump(pos, fw)
    
cap.release()
cv2.destroyAllWindows()