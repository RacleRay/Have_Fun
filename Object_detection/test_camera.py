import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

# 使用摄像头调用类
cap = cv2.VideoCapture(0)

PATH_TO_CKPT = r'D:\jupyter_code\Deep_Learning\Project\Diverse_Project\DeepInterest\Object_detection\ssd_mobilenet_v1_coco_2017_11_17\frozen_inference_graph.pb'
PATH_TO_LABELS = r'D:\jupyter_code\Deep_Learning\Project\Diverse_Project\DeepInterest\Object_detection\ssd_mobilenet_v1_coco_2017_11_17\mscoco_label_map.pbtxt'
NUM_CLASSES = 90


# 加载计算图
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()  # 储存读取的图数据
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name='')


# label处理
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        while True:
            res, image_np = cap.read()

            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # 转为RGB
            image_np_expanded = np.expand_dims(image_np, axis=0)  # dim = 4
            (boxes, scores, classes, num) = sess.run(
	    		[detection_boxes, detection_scores, detection_classes, num_detections],
	    		feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                               np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32),
                                                               np.squeeze(scores),
                                                               category_index,
                                                               use_normalized_coordinates=True,
                                                               line_thickness=8)

            # 显示结果
            cv2.imshow('object detection', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            # 退出，位运算求得ord（‘q’）
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break