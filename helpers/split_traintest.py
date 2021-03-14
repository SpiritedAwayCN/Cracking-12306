import os
import random

TRAIN_LABEL_OUTPUT = '../metadata/train_label_all.txt'
# TEST_LABEL_OUTPUT = 'metadata/test_label.txt'
SPLIT_RATIO = 0
DATA_DIR = '../archive'


with open(TRAIN_LABEL_OUTPUT, "w") as ftrain:

    for cate_name in os.listdir(DATA_DIR):
        file_list = os.listdir(os.path.join(DATA_DIR, cate_name))
        random.shuffle(file_list)

        for i in range(len(file_list)):
            ftrain.write(f'{cate_name}/{file_list[i]} {cate_name}\n')