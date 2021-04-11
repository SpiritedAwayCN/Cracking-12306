import os
import cv2
import random
import numpy as np
from tqdm import tqdm

ROOT_DIR = 'text/net/tagged_imgs_org/'
train_test_split = 0.8 # 0.8 for training

def cropping(img):
    img = 255 - img
    sumC = np.sum(img, axis=0) # 57
    sumR = np.sum(img, axis=1) # 19

    col = [i for i, val in enumerate(sumC) if val > 48]
    row = [i for i, val in enumerate(sumR) if val > 48]
    minX, maxX = min(col), max(col)+1
    minY, maxY = min(row), max(row)+1

    # output_img = np.zeros((19, 57))
    # nx, ny = maxX-minX, maxY-minY
    # sx, sy = (output_img.shape[1] - nx) // 2, (output_img.shape[0] - ny) // 2
    output_img = cv2.resize(img[minY:maxY, minX:maxX], (57, 57))

    return output_img

def main():
    texts = None
    labels = None
    texts_test = None
    labels_test = None

    for cate in tqdm(os.listdir(ROOT_DIR)):
        cate_dir = os.path.join(ROOT_DIR, cate)
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
            img = cropping(img)
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
            # cv2.imwrite(os.path.join(out_dir, filename), output_img)
        # break
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
    print(texts.shape, labels.shape)
    np.savez('textimg_trainV2.npz', texts, labels)
    if train_test_split < 1:
        print(texts_test.shape, labels_test.shape)
        np.savez('textimg_testV2.npz', texts_test, labels_test)

if __name__=='__main__':
    main()