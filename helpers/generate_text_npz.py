# WARNING: this code is so stupid!
# Too many extrados and nonsense np.vstack() and 
# memcpy operations were abused here

import enum
import numpy as np
import cv2
import os
import random
from tqdm import tqdm

DATA_DIR = 'archive_text/'
train_test_split = 0.8 # 0.8 for training

texts = None
labels = None
texts_test = None
labels_test = None

for cate in tqdm(os.listdir(DATA_DIR)):
    cate_dir = os.path.join(DATA_DIR, cate)
    last_img_train = None
    last_img_test = None

    ldir = os.listdir(cate_dir)
    random.shuffle(ldir)
    training_num = int(len(ldir) * train_test_split)
    test_num = len(ldir) - training_num

    label_train = np.full((training_num,), int(cate))
    label_test = np.full((test_num,), int(cate))
    for i, file_name in enumerate(ldir):
        img = cv2.imread(os.path.join(cate_dir, file_name), cv2.IMREAD_GRAYSCALE)
        img = img.reshape((1, *img.shape))
        if i < training_num:
            if not last_img_train is None:
                last_img_train = np.vstack((last_img_train, img))
                del img
            else:
                last_img_train = img
        else:
            if not last_img_test is None:
                last_img_test = np.vstack((last_img_test, img))
                del img
            else:
                last_img_test = img
    
    if not texts is None:
        texts = np.vstack((texts, last_img_train))
        # print(labels.shape, label.shape)
        labels = np.append(labels, label_train)
        del last_img_train
        del label_train
    else:
        texts = last_img_train
        labels = label_train

    if train_test_split < 1:
        if not texts_test is None:
            texts_test = np.vstack((texts_test, last_img_test))
            # print(labels.shape, label.shape)
            labels_test = np.append(labels_test, label_test)
            del last_img_test
            del label_test
        else:
            texts_test = last_img_test
            labels_test = label_test
    # break

print(texts.shape, labels.shape)
np.savez('text_img_train.npz', texts, labels)
if train_test_split < 1:
    print(texts_test.shape, labels_test.shape)
    np.savez('text_img_test.npz', texts_test, labels_test)

# (16906, 19, 57) (16906,)
# (4273, 19, 57) (4273,)