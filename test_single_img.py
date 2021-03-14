import numpy as np
from numpy.lib.npyio import load
import tensorflow as tf

import constants as c
from utils.data import load_image_multicrop_for_predict
from models.ResNetV2 import ResNetv2

_model = None
label_dict = [0] * c.num_class
def load_model_and_label(model_path='h5/20210314-02/ResNetV2-97.h5', label_path='metadata/label_to_content.txt'):
    global _model
    print(model_path)

    if not _model is None:
        return
    _model = ResNetv2()
    _model.build((None, ) + c.input_shape)
    _model.load_weights(model_path)
    with open(label_path, encoding='utf-8') as f:
        for line in f.readlines():
            id, name = line.strip().split(' ')
            label_dict[int(id)] = name

def pred_single(img):
    imgs = load_image_multicrop_for_predict(img)
    prediction = _model(imgs, training=False)
    prediction = tf.reduce_mean(prediction, axis=0)
    return prediction
