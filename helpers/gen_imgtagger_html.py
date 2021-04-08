import os

label_dict = [0] * 80
with open('../metadata/label_to_content.txt', encoding='utf-8') as f:
    for line in f.readlines():
        id, name, _ = line.strip().split(' ')
        label_dict[int(id)] = name.encode('gbk')

indexes = list(range(80))
indexes = sorted(indexes, key=lambda x: label_dict[x])

for idx in indexes:
    print(f'<option value="{idx}">{label_dict[idx].decode("gbk")}</option>')