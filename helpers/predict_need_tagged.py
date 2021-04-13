import os
import cv2
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from test_single_img import load_model_and_label, pred_single

load_model_and_label()

IMAGE_DIR = 'net/need_tagged/'
NPY_DIR = 'net/predictions/'

for file_name in tqdm(os.listdir(IMAGE_DIR)):
    img = cv2.imread(os.path.join(IMAGE_DIR, file_name))
    prediction = pred_single(img)
    np.save(os.path.join(NPY_DIR, file_name[:-3]+'npy'), prediction)
