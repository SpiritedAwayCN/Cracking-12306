import requests
import cv2
import time
import base64
import json
import numpy as np
import os
import hashlib

SAVE_DIR = 'text_imgs/'
CAPTCHA_GET_URL = "https://kyfw.12306.cn/passport/captcha/captcha-image?login_site=E&module=login&rand=sjrand"
CAPTCHA_CHK_URL = "https://kyfw.12306.cn/passport/captcha/captcha-check"

request_headers = {
    "Accept": "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9,en-GB;q=0.8,en;q=0.7",
    "Access-Control-Allow-Origin": "*",
    "Connection": "keep-alive",
    "Host": "kyfw.12306.cn",
    "Referer": "https://kyfw.12306.cn/otn/resources/login.html",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origi",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest"
}

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

session = requests.session()
session.headers = request_headers

def extract_text_img(img):
    if img is None:
        return None
    text_img = img[:29, 117:250]
    text_img = cv2.cvtColor(text_img, cv2.COLOR_BGR2GRAY)
    
    text_img2 = cv2.GaussianBlur(text_img, (3,3), 1)
    sobelY = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    edges = cv2.filter2D(text_img2, -1, sobelY)
    _, thres = cv2.threshold(edges, 0, 1, cv2.THRESH_OTSU)
    
    col_sum = np.sum(thres, axis=0)
    divides = np.argwhere(col_sum > 27).squeeze()
    imgs = []
    
    if len(divides) < 2:
        _, thres = cv2.threshold(text_img, 0, 1, cv2.THRESH_OTSU)
        col_sum = np.sum(thres, axis=0)
        divides = np.argwhere(col_sum < 26)
        right = np.max(divides) + 12
        imgs.append(text_img[:, :right])
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
        
        imgs.append(text_img[:, :ans[0]].copy())
        imgs.append(text_img[:, ans[0]:ans[1]].copy())
        # assert len(ans) == 2
    return imgs

def fetch_img():
    timestamp = int(round(time.time()*1000))
    response = session.get(CAPTCHA_GET_URL + f'&{timestamp}')
    captcha = response.content

    bytestream = np.array(bytearray(captcha))
    img = cv2.imdecode(bytestream, cv2.IMREAD_COLOR)
    return img

def send_fake_request():
    paras = {
        "answer":'104,110,264,124', 
        "rand": "sjrand",
        "login_site": "E"
    }

    response = session.get(CAPTCHA_CHK_URL, params=paras)
    # print(response.text)

def save_imgs(imgs):
    global sc, dc
    for img in imgs:
        hashlib.md5()
        m = hashlib.md5()
        for val in img:
            m.update(val)
        fname = f'{int(time.time())}-{m.hexdigest()}.png'
        cv2.imwrite(os.path.join(SAVE_DIR, fname), img)
        print('Saved:', fname)
    if len(imgs) == 1:
        sc += 1
    else:
        dc += 2

single_count = 10000
double_count = 10000
sc = 73
dc = 0

def main():
    status = 0
    while True:
        if status== 1 and ((sc > dc + 1000 and dc < double_count) or sc >= single_count):
            send_fake_request()
        img = fetch_img()
        imgs = extract_text_img(img)
        t = np.random.normal(5, 0.1)
        if not imgs is None:
            status = len(imgs)
            save_imgs(imgs)
            if sc >= single_count and dc >= double_count:
                break
        else:
            print('Skipped due to Nonetype, sleeping 10 s')
            t += 10
        time.sleep(t)

if __name__ == '__main__':
    main()