import numpy as np
import tensorflow as tf
import cv2

import constants as c
from utils.data import load_image_multicrop_for_predict
from models.ResNetV2 import *
from models.ResNet_SE import *
from models.ResNetV2_text import ResNetv2_text

_model = None
_text_model = None
label_dict = [0] * c.num_class
def load_model_and_label(text_model_path='h5/final/text.h5',
    model_path='h5/20210413-03/ResNetV2-50.h5',
    label_path='metadata/label_to_content.txt'):
    global _model, _text_model
    print('image recognition model:', model_path)
    print('text recognition model:', text_model_path)

    if not _model is None:
        return
    
    if model_path == 'boosting':
        _model = [ResNetSEM1(), ResNetv2M2(), ResNetSEM3(), ResNetv2M4(), ResNetv2M4()]
        _model_path = [f'h5/final/M{i}.h5' for i in range(1, 6)]
        for model, path in zip(_model, _model_path):
            model.build((None, ) + c.input_shape)
            model.load_weights(path)
    else:
        _model = ResNetv2()
        _model.build((None, ) + c.input_shape)
        _model.load_weights(model_path)

    _text_model = ResNetv2_text(c.num_class)
    _text_model.build((None, 57, 57, 1))
    _text_model.load_weights(text_model_path)
    with open(label_path, encoding='utf-8') as f:
        for line in f.readlines():
            id, name, _ = line.strip().split(' ')
            label_dict[int(id)] = name

def pred_single(img, boosting=False):
    imgs = load_image_multicrop_for_predict(img)
    if boosting:
        prediction = [model(imgs, training=False) for model in _model]
        prediction = tf.concat(prediction, axis=0)
    else:
        prediction = _model(imgs, training=False)
    prediction = tf.reduce_mean(prediction, axis=0)
    return prediction

def pred_text(img):
    text_imgs = extract_text_img(img)
    if text_imgs is None:
        return None
    text_imgs = tuple(map(cropping, text_imgs))
    text_imgs = np.vstack(text_imgs).astype(np.float32)
    text_imgs = (text_imgs - 74.19824782421746) / 59.435693003198594
    text_prediction = _text_model(text_imgs, training=False)
    return text_prediction


def extract_text_img(img):
    text_img = img[:29, 117:250]
    text_img = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    
    #text_img2 = cv2.GaussianBlur(text_img, (3,3), 1)
    text_img2 = text_img
    sobelY = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    edges = cv2.filter2D(text_img2, cv2.CV_32F, sobelY)
    edges = np.abs(edges).clip(0, 255).astype(np.uint8)
    _, thres = cv2.threshold(edges, 0, 1, cv2.THRESH_OTSU)
    
    col_sum = np.sum(thres, axis=0)
    divides = np.argwhere(col_sum > 27).flatten()
    imgs = []
    
    if len(divides) < 2:
        _, thres = cv2.threshold(text_img, 0, 1, cv2.THRESH_OTSU)
        col_sum = np.sum(thres, axis=0)
        divides = np.argwhere(col_sum < 26)
        right = np.max(divides) + 12
        imgs.append(text_img[3:22, :right])
    else:
        last_num = 2
        cnt = 0
        sums = 0
        ans = []
        for num in divides:
            if num == last_num + 1:
                sums += num
                cnt += 1
            else:
                if cnt > 0:
                    ans.append(int(round(sums / cnt)))
                sums = num
                cnt = 1
            last_num = num
        ans.append(int(round(sums / cnt)))
        if(len(ans) < 2):
            return None # error!

        imgs.append(text_img[3:22, :ans[0]].copy())
        imgs.append(text_img[3:22, ans[0]:ans[1]].copy())
        # assert len(ans) == 2
    return imgs

def cropping(img):
    img = 255 - img
    sumC = np.sum(img, axis=0) # 57
    sumR = np.sum(img, axis=1) # 19

    col = [i for i, val in enumerate(sumC) if val > 48]
    row = [i for i, val in enumerate(sumR) if val > 48]
    minX, maxX = min(col), max(col)+1
    minY, maxY = min(row), max(row)+1

    # output_img = np.zeros((19, 57))
    # nx, ny = maxX-minX, maxY-minY
    # sx, sy = (output_img.shape[1] - nx) // 2, (output_img.shape[0] - ny) // 2
    output_img = cv2.resize(img[minY:maxY, minX:maxX], (57, 57))
    return output_img.reshape(1, 57, 57, 1)