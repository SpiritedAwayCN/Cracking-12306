import os
import pathlib

DATA_DIR = './archive'
counter = 0

with open('label_to_content.txt', "w") as f:
    for dirname in os.listdir(DATA_DIR):
        f.write('%02d %s\n'%(counter, dirname))
        old_path = os.path.join(DATA_DIR, dirname)
        new_path = os.path.join(DATA_DIR, "%02d"%(counter))
        os.rename(old_path, new_path)
        counter += 1
