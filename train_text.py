import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAvgPool2D, Dense, Activation, AvgPool2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

from models.ResNetV2_text import ResNetv2_text

num_classes = 80
input_shape = 57, 57, 1

batch_size = 128
total_epoches = 50
iterations_per_epoch = (16906 - 1 + batch_size) // batch_size
lamb = 1e-3

# log_dir = 'cifar10'

train_data = np.load('metadata/textimg_trainV2.npz')
test_data = np.load('metadata/textimg_testV2.npz')
# print(train_data.files)
x_train, y_train = train_data['arr_0'], train_data['arr_1']
x_test, y_test = test_data['arr_0'], test_data['arr_1']
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

mean = np.mean(x_train)
std = np.std(x_train)
print(mean, std)
# 74.19824782421746 59.435693003198594

zero_val = -mean / std

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

print(x_train.shape, x_test.shape)

model = ResNetv2_text(num_classes, lamb=lamb)
model.build((None, ) + input_shape)

learning_rate_schedules = keras.experimental.CosineDecay(initial_learning_rate=0.05,decay_steps=total_epoches * iterations_per_epoch, alpha=0.0001)
optimizer = keras.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9, nesterov=True)

datagen = ImageDataGenerator(
    width_shift_range=0.125, 
    height_shift_range=0.125,
    fill_mode='constant',
    cval=zero_val)

# datagen.fit(x_train)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

cbks = [
    # TensorBoard(log_dir=os.path.join(log_dir), histogram_freq=0),
    ModelCheckpoint('h5/20210411-text01/checkpoint-{epoch}.h5', save_best_only=False, mode='auto')
    ]

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=iterations_per_epoch,
    epochs=total_epoches,
    callbacks=cbks,
    validation_data=(x_test, y_test))