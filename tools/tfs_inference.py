import requests,json
from PIL import Image
import numpy as np
import tensorflow as tf
from absl.flags import FLAGS

def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}

def container_predict(image_file_path):
    
    file_name = image_file_path

    img_raw = tf.image.decode_image(open(file_name, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img,416)

    url = 'http://localhost:8501/v1/models/default:predict'
    data = json.dumps({'instances':img.numpy().tolist()})

    r =requests.post(url,data)
    r = json.loads(r.text)
    print(r)
    boxes, scores, classes, nums = r['predictions'][0]["yolo_nms"], r['predictions'][0][
            "yolo_nms_1"], r['predictions'][0]["yolo_nms_2"], r['predictions'][0]["yolo_nms_3"]

if __name__ == '__main__':
    image_file_path = './data/street.jpg'
    try:
        container_predict(image_file_path)
    except SystemExit:
        pass