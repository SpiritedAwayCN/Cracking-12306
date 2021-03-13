import os
import pathlib

err = '妗ｆ琚_'
path = '23'

if not os.path.exists(path):
    os.mkdir(path)

sum = 0
for item in pathlib.Path('./').glob(f'{err}*.png'):
    # print(item.name.replace(err, ''))
    item.rename(os.path.join(path, item.name.replace(err, '')))
    sum += 1

print(f'{path}:', sum)
