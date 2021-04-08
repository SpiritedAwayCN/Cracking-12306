from flask import *
from flask_compress import Compress
import requests
import cv2
import io
import os
import time
import random
import numpy as np
import base64
import shutil
import numpy as np

app = Flask(__name__)
Compress(app)
app.secret_key = 'dfjdignrkngrip'

app.debug = True

PATHNAME_IN = 'untagged_imgs2'
PATHNAME_OUT = 'tagged_imgs'
PATHNAME_SKIP = 'skipped'

for path in (PATHNAME_IN, PATHNAME_OUT, PATHNAME_SKIP):
    if not os.path.exists(path):
        os.mkdir(path)

piny_dict = {}
label_dict = [0] * 80
with open('../../metadata/label_to_content.txt', encoding='utf-8') as f:
    for line in f.readlines():
        id, name, py = line.strip().split(' ')
        label_dict[int(id)] = name
        piny_dict[py] = int(id)
label_dict = np.array(label_dict)
# print(len(piny_dict.keys()))
assert len(piny_dict.keys()) == 80

@app.before_request
def setup_session():
    if 'count' not in session:
        session['count'] = 0

def return_img_stream(img_local_path):
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    return img_stream

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/captcha_frame')
def captcha_frame():
    if 'serial' not in session or not os.path.isfile(f'{PATHNAME_IN}/{session["serial"]}.png'):
        session['serial'] = os.listdir(PATHNAME_IN)[0].partition('.')[0]
        
    img_path = f'{PATHNAME_IN}/{session["serial"]}.png'
    img_stream = return_img_stream(img_path).decode()


    return render_template('captcha_frame.html', img_stream=img_stream)

@app.route('/submit_captcha', methods=['POST'])
def submit_captcha():
    assert 'serial' in session
    assert '/' not in session['serial']
    serial = session['serial']
    captcha = request.form['captcha'].lower()

    if captcha == '-':
        shutil.move('%s/%s.png'%(PATHNAME_IN, serial), '%s/%s.png'%(PATHNAME_SKIP, serial))
        flash(f'已暂时忽略{serial}，存入暂存文件夹')
        del session['serial']
        return redirect(url_for('index'))
    if captcha not in piny_dict.keys():
        flash('未检测到对应类别')
        return redirect(url_for('index'))

    cate_id = piny_dict[captcha]

    flash(f'已成功标记类别：{label_dict[cate_id]}')
    session['count'] += 1

    dst_dir = os.path.join(PATHNAME_OUT, "%02d"%(cate_id))
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    shutil.move('%s/%s.png'%(PATHNAME_IN, serial), '%s/%s.png'%(dst_dir, serial))

    del session['serial']
    return redirect(url_for('index'))

app.run('0.0.0.0', 10192)