import requests
import time
import json
import base64
import os

import cv2
import numpy as np
import tensorflow as tf

from test_single_img import load_model_and_label, pred_single, label_dict

_url = 'https://kyfw.12306.cn/passport/captcha/captcha-image64'
IMAGE_SAVE_DIR = 'net/need_tagged'
RES_SAVE_DIR = 'net/predictions'

def get_captcha_img():
    r = requests.get(_url)
    img_str = json.loads(r.content)['image']

    decoded = base64.b64decode(img_str)
    img = np.asarray(bytearray(decoded), dtype="uint8")
    return cv2.imdecode(img, cv2.IMREAD_COLOR)


def gen_imgs(img):
    interval = 6
    length = 66
    for x in range(41, img.shape[0] - length, interval + length):
        for y in range(interval, img.shape[1] - length, interval + length):
            yield img[x:x + length, y:y + length]

def main():
    counter = 0
    load_model_and_label()
    while True:
        sleep_sec = max(np.random.normal(6, 1), 0)
        time.sleep(sleep_sec)
        counter += 1
        if counter > 1000:
            break
        if counter % 20 == 0:
            print('counter = ', counter)
        img = get_captcha_img()
        generator = gen_imgs(img)
        for i, image in enumerate(generator):

            prediction = pred_single(image).numpy()
            max_val, id = max(prediction), np.argmax(prediction)
            if max_val > 0.55:
                continue
            file_name = '{:07d}-{}'.format(counter, i)
            print('Saving {} with max probobility {:.2f} of {}'.format(file_name, max_val, label_dict[id]))
            np.save(os.path.join(RES_SAVE_DIR, file_name), prediction)
            cv2.imwrite(os.path.join(IMAGE_SAVE_DIR, file_name + '.png'), image)
        # break



if __name__ == '__main__':
    main()