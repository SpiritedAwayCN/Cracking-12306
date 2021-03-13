import numpy as np
import os
import cv2
from tqdm import tqdm

DATASET_DIR = 'archive'

EX = np.zeros((3,))
EXY = np.zeros((3, 3))
s = 0

for cate_name in tqdm(os.listdir(DATASET_DIR)):
    cate_dir = os.path.join(DATASET_DIR, cate_name)

    for file_name in os.listdir(cate_dir):
        img = np.fromfile(os.path.join(cate_dir, file_name), dtype=np.uint8)
        img = cv2.imdecode(img, -1).astype(np.float32)
        img = img.reshape((-1, 3))

        mean = np.average(img, axis=0)
        cov = np.dot(img.T, img / img.shape[0])
        s += 1
        EX = (s - 1) / s * EX + mean / s
        EXY = (s - 1) / s * EXY + cov / s
    # break

print('mean = ', EX)
# print('EXY = ', EXY)

cov = EXY - np.outer(EX, EX)
# print('cov = ', cov)
std = np.sqrt(np.diagonal(cov))
print('std = ', std)

S, U = np.linalg.eig(cov)  # not ordered
print('eigval = ', S)
print('eigvec = ', U)