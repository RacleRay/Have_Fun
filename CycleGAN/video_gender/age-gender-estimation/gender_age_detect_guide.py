# -*- coding: utf-8 -*-

from wide_resnet import WideResNet
import numpy as np
import cv2
import dlib

depth = 16
width = 8
img_size = 64
# 人脸性别年龄预测模型
model = WideResNet(img_size, depth=depth, k=width)()
model.load_weights('weights.hdf5')


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    # 在视频中写入预测结果
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


# dilb读取video，检测人脸
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, image_np = cap.read()
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    img_h = image_np.shape[0]
    img_w = image_np.shape[1]

    detected = detector(image_np, 1)
    faces = []

    # 检测出人脸
    if len(detected) > 0:
        for i, d in enumerate(detected):
            x0, y0, x1, y1, w, h = d.left(), d.top(), d.right(), d.bottom(), d.width(), d.height()
            cv2.rectangle(image_np, (x0, y0), (x1, y1), (255, 0, 0), 2)

            # 扩大检测范围
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
            # 64 * 64
            faces.append(cv2.resize(image_np[y0: y1, x0: x1, :], (img_size, img_size)))

        faces = np.array(faces)

        results = model.predict(faces)
        predicted_genders = results[0]
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = results[1].dot(ages).flatten()

        for i, d in enumerate(detected):
            label = '{}, {}'.format(int(predicted_ages[i]), 'F' if predicted_genders[i][0] > 0.5 else 'M')
            draw_label(image_np, (d.left(), d.top()), label)

    cv2.imshow('gender and age', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break