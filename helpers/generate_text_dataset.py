import os
import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR = 'text/net/tagged_imgs_org/'
OUT_DIR = 'archive_text/'

def cropping(img):
    img = 255 - img
    sumC = np.sum(img, axis=0) # 57
    sumR = np.sum(img, axis=1) # 19

    col = [i for i, val in enumerate(sumC) if val > 48]
    row = [i for i, val in enumerate(sumR) if val > 48]
    minX, maxX = min(col), max(col)+1
    minY, maxY = min(row), max(row)+1

    output_img = np.zeros((19, 57))
    nx, ny = maxX-minX, maxY-minY
    sx, sy = (output_img.shape[1] - nx) // 2, (output_img.shape[0] - ny) // 2
    output_img[sy: sy+ny, sx: sx+nx] = img[minY:maxY, minX:maxX]

    return output_img

def main():
    for cate_dir in tqdm(os.listdir(ROOT_DIR)):
        out_dir = os.path.join(OUT_DIR, cate_dir)
        cate_dir = os.path.join(ROOT_DIR, cate_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        for filename in os.listdir(cate_dir):
            img = cv2.imread(os.path.join(cate_dir, filename), cv2.IMREAD_GRAYSCALE)
            output_img = cropping(img)
            cv2.imwrite(os.path.join(out_dir, filename), output_img)
        # break

if __name__=='__main__':
    main()