import os
import random

DATA_DIR = 'archive'

fopen_list = [open(f'metadata/test_label{i}.txt', 'w') for i in range(1, 6)]

for cate_name in os.listdir(DATA_DIR):
    file_list = os.listdir(os.path.join(DATA_DIR, cate_name))
    random.shuffle(file_list)

    counts = [len(file_list) // 5] * 5

    indexes = [0, 1, 2, 3, 4]
    random.shuffle(indexes)

    for i in range(len(file_list) % 5):
        counts[indexes[i]] += 1
    
    st = 0
    for i in range(5):
        for j in range(counts[i]):
            fopen_list[i].write(f'{cate_name}/{file_list[st + j]} {cate_name}\n')
        st += j
