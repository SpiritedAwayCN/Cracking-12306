f1 = open('original_label.txt', encoding='utf-8')
f2 = open('../metadata/label_to_content.txt', encoding='utf-8')

org_dict = {}
for i, label in enumerate(f2.readlines()):
    label = label.split(' ')[1].strip()
    org_dict[label] = i

res_map = [0] * 80
for i, label in enumerate(f1.readlines()):
    res_map[i] = org_dict[label.strip()]
print(res_map)

f1.close()
f2.close()

# [15, 64, 65, 26, 48, 12, 27, 41, 22, 54, 9, 79, 45, 17, 8, 30, 44, 78, 34, 33, 69, 66, 28, 29, 2, 25, 4, 35, 51, 77, 39, 47, 31, 76, 62, 3, 63, 19, 71, 46, 
# 50, 38, 43, 68, 75, 55, 13, 40, 1, 24, 42, 36, 58, 60, 53, 7, 52, 11, 23, 18, 5, 70, 16, 14, 73, 20, 67, 49, 0, 61, 6, 32, 72, 56, 37, 74, 57, 59, 21, 10]