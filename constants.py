input_shape = 56, 56, 3
num_class = 80

total_epoches = 100
batch_size = 64

train_num = 12752
val_num = 0

iterations_per_epoch = train_num // batch_size + 1
test_iterations = val_num // batch_size + 1

weight_decay = 5e-4
label_smoothing = 0.1


'''
numeric characteristics
'''
mean =  [153.9959882, 163.42529556, 174.66020828]
std =  [88.32422872, 82.59181539, 78.60014756]

# eigval =  [18793.85624672, 1592.25590705, 360.43236465]
eigval = [137.30433048,  39.85269598,  18.96955458]
eigvec =  [[-0.61348107, -0.62605035,  0.48135427], 
 [-0.59102031, -0.04028949, -0.80564989], 
 [-0.52377092,  0.77874111,  0.34529163]]

 