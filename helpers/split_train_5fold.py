# read in and merge
metadata = set()
for i in range(1,6):
    with open(f'metadata/test_label{i}.txt') as f:
        metadata |= set([s[:-1] for s in f.readlines()])
metadata -= {''}

print(f'{len(metadata)} data i total, splitting into 5 folds.')

#split and write out
for i in range(1,6):
    with open(f'metadata/test_label{i}.txt') as f:
        train = metadata - set([s[:-1] for s in f.readlines()])
    with open(f'metadata/train_label{i}.txt','w') as f:
        train = sorted(list(train))
        f.write('\n'.join(train))
