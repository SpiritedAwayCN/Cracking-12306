import json
import time

import requests
import hashlib
import json
import base64

import cv2
import os
import numpy as np

from test_single_img import load_model_and_label, pred_single, label_dict, pred_text

CAPTCHA_GET_URL = "https://kyfw.12306.cn/passport/captcha/captcha-image?login_site=E&module=login&rand=sjrand"
CAPTCHA_CHK_URL = "https://kyfw.12306.cn/passport/captcha/captcha-check"

CAPTCHA_OUTPUT_DIR = 'net/bootstrap_failed'
PROCESS_OUTPUT_DIR = 'net/preprocess_failed'
IMAGE_OUTPUT_DIR = 'net/need_tagged'

if not os.path.exists(CAPTCHA_OUTPUT_DIR):
    os.mkdir(CAPTCHA_OUTPUT_DIR)
if not os.path.exists(IMAGE_OUTPUT_DIR):
    os.mkdir(IMAGE_OUTPUT_DIR)
if not os.path.exists(PROCESS_OUTPUT_DIR):
    os.mkdir(PROCESS_OUTPUT_DIR)

request_headers = {
    "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9,en-GB;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Host": "kyfw.12306.cn",
    "Referer": "https://kyfw.12306.cn/otn/resources/login.html",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origi",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}

load_model_and_label()

def get_new_session():
    session = requests.session()
    session.headers = request_headers
    return session

def gen_imgs(img):
    interval = 6
    length = 66
    for x in range(41, img.shape[0] - length, interval + length):
        for y in range(interval, img.shape[1] - length, interval + length):
            yield img[x:x + length, y:y + length]

def fetch_captcha(session):
    timestamp = int(round(time.time()*1000))
    response = session.get(CAPTCHA_GET_URL + f'&{timestamp}')
    captcha = response.content

    bytestream = np.array(bytearray(captcha))
    img = cv2.imdecode(bytestream, cv2.IMREAD_COLOR)
    return img

def predict_captcha(img, md5_str, save=True):
    top_k = 5

    selected_index = []

    text_prediction = pred_text(img)
    if text_prediction is None:
        return 0, 0
    text_indexes = []

    for i in range(text_prediction.shape[0]):
        index_list = np.argpartition(text_prediction[i], -top_k)[-top_k:]
        index_list = sorted(index_list, key=lambda x: -text_prediction[i, x].numpy())
        # for idx in index_list:
        #     print("%s%.4f"%(label_dict[idx], text_prediction[i, idx]), end=', ')
        text_indexes.append(index_list[0])
    #     print("")
    # print("------------------")
    # label_np = np.array(label_dict)
    for i, image in enumerate(gen_imgs(img)):
        prediction = pred_single(image)
        
        index_list = np.argpartition(prediction, -top_k)[-top_k:]
        index_list = sorted(index_list, key=lambda x: -prediction[x].numpy())
        if index_list[0] in text_indexes:
            selected_index.append(i)
        elif index_list[0] in (18, 69) and (18 in text_indexes or 69 in text_indexes):
            selected_index.append(i) # 钟表和挂钟都算
        if save and prediction[index_list[0]] < 0.55:
            cv2.imwrite(os.path.join(IMAGE_OUTPUT_DIR, f'{md5_str}-im{i}.png'), image)

        # labels = label_np[index_list][::-1]

    return selected_index, text_prediction.shape[0]

def send_check_request(session, selected_index):
    indices = {0: "48,70",
           1: "100,70",
           2: "180,70",
           3: "250,70",
           4: "48,150",
           5: "100,150",
           6: "180,150",
           7: "250,150"}
    ans = ','.join([indices[i] for i in selected_index])
    # print(ans)
    paras = {
        "answer": ans, 
        "rand": "sjrand",
        "login_site": "E"
    }
    response = session.get(CAPTCHA_CHK_URL, params=paras)
    response_dict = json.loads(response.text)
    try:
        status = int(response_dict["result_code"])
    except ValueError:
        status = 6
    if status == 4:
        return 1, response_dict
    elif status == 5 or status == 8:
        return -1, response_dict
    else:
        return 0, response_dict

def main():
    crawl_count = 0
    total = [0, 0]
    succ = [0, 0]
    session = None

    while True:
        if crawl_count % 50 == 0:
            print('getting new session...')
            if not session is None:
                session.close()
            t = np.random.normal(5, 0.1)
            time.sleep(t)
            session = get_new_session()
        img = fetch_captcha(session)

        if img is None:
            print('got Nonetype image, sleeping 10s and reopening session...')
            if not session is None:
                session.close()
            time.sleep(10)
            session = get_new_session()
            crawl_count += 1
            continue
    
        m = hashlib.md5()
        for val in img:
            m.update(val)
        md5_str = m.hexdigest()
        
        fname = f'{int(time.time())}-{m.hexdigest()}.png'
        selected_index, text_num = predict_captcha(img, md5_str)
        if text_num == 0:
            print('text preprocessing failed!')
            cv2.imwrite(os.path.join(PROCESS_OUTPUT_DIR, fname), img)
            time.sleep(5)
            crawl_count += 1
            continue

        code, msg = send_check_request(session, selected_index)
        if code == 1:
            succ[text_num - 1] += 1
        if code == -1:
            cv2.imwrite(os.path.join(CAPTCHA_OUTPUT_DIR, fname), img)
        if code != 0:
            total[text_num - 1] += 1
            acc0 = succ[0]/total[0] if total[0]!=0 else 0
            acc1 = succ[1]/total[1] if total[1]!=0 else 0
            acc = (succ[0]+succ[1]) / (total[0]+total[1])
            print("loop %d: acc(1) %d/%d=%.4f"%(crawl_count+1, succ[0], total[0], acc0), end=', ')
            print('acc(2) %d/%d=%.4f, total_acc=%.4f'%(succ[1], total[1], acc1, acc))
        else:
            print(msg)

        crawl_count += 1
        t = np.random.normal(5, 0.1)
        time.sleep(t)

if __name__=='__main__':
    main()