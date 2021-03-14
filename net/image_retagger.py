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

PATHNAME_IN = 'need_tagged'
PATHNAME_IN2 = 'predictions'
PATHNAME_OUT = 'retagged'
PATHNAME_SKIP = 'skipped'

for path in (PATHNAME_IN, PATHNAME_IN2, PATHNAME_OUT, PATHNAME_SKIP):
    if not os.path.exists(path):
        os.mkdir(path)

top_k = 5

label_dict = [0] * 80
with open('../metadata/label_to_content.txt', encoding='utf-8') as f:
    for line in f.readlines():
        id, name = line.strip().split(' ')
        label_dict[int(id)] = name
label_dict = np.array(label_dict)

@app.before_request
def setup_session():
    if 'count' not in session:
        session['count'] = 0
    if 'correct' not in session:
        session['correct'] = 0

def return_img_stream(img_local_path):
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    return img_stream

@app.route('/')
def index():
    if 'serial' not in session or not os.path.isfile(session['serial']):
        session['serial'] = os.listdir(PATHNAME_IN)[0].partition('.')[0]
    
    predict = np.load(f'{PATHNAME_IN2}/{session["serial"]}.npy')
    session['predict'] = int(np.argmax(predict))
    
    index_list = np.argpartition(predict, -top_k)[-top_k:]
    index_list = sorted(index_list, key=lambda x: -predict[x])

    top_label_dict = {}
    for cate_id in index_list:
        top_label_dict[label_dict[cate_id]] = "{:02d}".format(cate_id)
    
    return render_template('index.html', top_label_dict=top_label_dict)

@app.route('/captcha_frame')
def captcha_frame():
    assert 'serial' in session
    assert '/' not in session['serial']

    img_path = f'{PATHNAME_IN}/{session["serial"]}.png'
    img_stream = return_img_stream(img_path).decode()

    predict = np.load(f'{PATHNAME_IN2}/{session["serial"]}.npy')
    index_list = np.argpartition(predict, -top_k)[-top_k:]
    index_list = sorted(index_list, key=lambda x: -predict[x])
    
    top_label_dict = {}
    for cate_id in index_list:
        top_label_dict[label_dict[cate_id]] = predict[cate_id]

    return render_template('captcha_frame.html', img_stream=img_stream, top_label_dict=top_label_dict)

@app.route('/submit_captcha', methods=['POST'])
def submit_captcha():
    assert 'serial' in session
    assert '/' not in session['serial']
    serial = session['serial']
    tagged_id = request.form['cate_id']

    if tagged_id == 'marked':
        os.remove('%s/%s.png'%(PATHNAME_IN, serial))
        os.remove('%s/%s.npy'%(PATHNAME_IN2, serial))
        flash('已删除')
        del session['serial']
        return redirect(url_for('index'))

    if tagged_id == 'skipped':
        shutil.move('%s/%s.png'%(PATHNAME_IN, serial), '%s/%s.png'%(PATHNAME_SKIP, serial))
        flash(f'已暂时忽略{serial}，存入暂存文件夹')
        del session['serial']
        return redirect(url_for('index'))

    cate_id = int(tagged_id)
    assert 0 <= cate_id and cate_id < 80

    flash(f'已成功标记类别：{label_dict[cate_id]}')
    session['count'] += 1
    if session['predict'] == cate_id:
        session['correct'] += 1

    dst_dir = os.path.join(PATHNAME_OUT, tagged_id)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    shutil.move('%s/%s.png'%(PATHNAME_IN, serial), '%s/%s.png'%(dst_dir, serial))
    os.remove('%s/%s.npy'%(PATHNAME_IN2, serial))

    del session['serial']
    return redirect(url_for('index'))

app.run('0.0.0.0', 10191)