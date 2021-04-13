import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import cv2

import constants as c
from utils import utils

def load_list(list_path, image_path):
    images = []
    labels = []
    with open(list_path, 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            images.append(os.path.join(image_path, line[0]))
            labels.append(int(line[1]))
    return images, labels

def data_augmentation(image):
    # 随机裁剪
    height, weight, _ = np.shape(image)
    crop_x = np.random.randint(height - c.input_shape[0])
    crop_y = np.random.randint(weight - c.input_shape[1])
    image = image[crop_x:crop_x + c.input_shape[0], crop_y:crop_y + c.input_shape[1]]

    # 随机水平翻转
    if(np.random.rand() < 0.5):
        image = cv2.flip(image, 1)

    # hsv上的偏移
    offset_h = np.random.uniform(-9, 9) #这个不能太过分
    offset_s = np.random.uniform(0.8, 1.25)
    offset_v = np.random.uniform(0.8, 1.25)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 0] = (image_hsv[:, :, 0] + offset_h) % 360.
    image_hsv[:, :, 1] = np.minimum(image_hsv[:, :, 1] * offset_s, 1.)
    image_hsv[:, :, 2] = np.minimum(image_hsv[:, :, 2] * offset_v, 255.)
    image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # pca噪音 复用了两个变量
    offset_h = np.random.normal(0, 0.05, size=(3,))
    offset_s = np.dot(c.eigvec * offset_h, c.eigval)
    image = np.maximum(np.minimum(image + offset_s, 255.), 0.)

    return image

def load_image(path, labels, augments=False):
    image = cv2.imread(path.numpy().decode()).astype(np.float32)

    if augments:
        image = data_augmentation(image)
    else:
        height, width, _ = np.shape(image)
        input_height, input_width, _ = c.input_shape
        crop_x = (width - input_width) // 2
        crop_y = (height - input_height) // 2
        image = image[crop_y: crop_y + input_height, crop_x: crop_x + input_width, :]
    
    # 可视化请注释以下部分
    for i in range(3):
        image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]

    label = keras.utils.to_categorical(labels, c.num_class)
    return image, label

def load_image_multicrop(path, labels):
    image = cv2.imread(path.numpy().decode()).astype(np.float32)

    images = utils.crop_ten(image)
    image = np.array(images, dtype=np.float32)

    for i in range(3):
        image[..., i] = (image[..., i] - c.mean[i]) / c.std[i]
    label = keras.utils.to_categorical(labels, c.num_class)
    return image, label

def load_image_multicrop_for_predict(image):
    image = image.astype(np.float32)

    images = utils.crop_ten(image)
    images = np.array(images, dtype=np.float32)

    for i in range(3):
        images[..., i] = (images[..., i] - c.mean[i]) / c.std[i]
    return images

def get_train_dataset(list_path="./metadata/train_label.txt"):
    images, labels = load_list(list_path, "./archive")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(len(images)).repeat()
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, True], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    return dataset

def get_test_dataset(list_path="./metadata/test_label.txt"):
    images, labels = load_list(list_path, "./archive")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image, inp=[x, y, False], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(c.batch_size)
    return dataset

def get_predict_dataset(list_path="./metadata/test_label.txt"):
    images, labels = load_list(list_path, "./archive")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: tf.py_function(load_image_multicrop, inp=[x, y], Tout=[tf.float32, tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def main():
    train_iter = get_train_dataset().__iter__()

    image, labels =train_iter.next()
    print(np.shape(image), np.shape(labels))

    for i in range(10):
        cv2.imshow('show', image[i].numpy().astype(np.uint8))
        cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
