input_shape = 56, 56, 3
num_class = 80

total_epoches = 50
batch_size = 64

train_num = 10652
val_num = 1142

iterations_per_epoch = train_num // batch_size + 1
test_iterations = val_num // batch_size + 1

weight_decay = 5e-4
label_smoothing = 0.1


'''
numeric characteristics
'''
mean =  [154.64720717, 163.98750114, 175.11027269]
std =  [88.22176357, 82.46385599, 78.50590683]

# eigval =  [18793.85624672, 1592.25590705, 360.43236465]
eigval = [137.09068621,  39.90308142,  18.98505635]
eigvec =  [[-0.61372719, -0.62390345,  0.48382169],
 [-0.59095847, -0.0433538, -0.80553618],
 [-0.52355231, 0.78029798, 0.34209362]]

 