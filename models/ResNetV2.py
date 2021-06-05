import tensorflow as tf
import constants as c
from tensorflow.keras.layers import Conv2D, GlobalAvgPool2D, BatchNormalization, Dense, Activation, AvgPool2D, MaxPooling2D

class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=(1, 1), **kwargs):
        self.strides = strides
        if strides != (1, 1):
            self.shortcut = Conv2D(filters, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')

        self.conv_0 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.conv_1 = Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')
        self.bn_0 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.bn_1 = BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.activation0 = Activation('relu')
        self.activation1 = Activation('relu')
        self.avgpool = AvgPool2D((2, 2), strides=(2, 2), padding='same')
        super(BasicBlock, self).__init__(**kwargs)

    def call(self, inputs, training):
        res = self.bn_0(inputs, training=training)
        res = self.activation0(res)

        if self.strides != (1, 1):
            shortcut = self.avgpool(res)
            shortcut = self.shortcut(shortcut)
        else:
            shortcut = inputs

        res = self.conv_0(res)
        res = self.bn_1(res, training=training)
        res = self.activation1(res)
        res = self.conv_1(res)

        output = res + shortcut
        return output

class ResNetv2(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(ResNetv2, self).__init__(**kwargs)

        self.conv0 = Conv2D(16, (7, 7), strides=(1, 1), name='conv0', padding='same', use_bias=False, kernel_initializer='he_normal')
        self.block_collector = []
        block_num = 6, 6, 6
        filters_num = 16, 32, 64
        for i in range(1, 4):
            if i == 1:
                self.block_collector.append(BasicBlock(filters_num[i-1], name='conv1_0'))
            else:
                self.block_collector.append(BasicBlock(filters_num[i-1], strides=(2, 2), name='conv{}_0'.format(i)))

            for j in range(1, block_num[i-1]):
                self.block_collector.append(BasicBlock(filters_num[i-1], name='conv{}_{}'.format(i, j)))

        self.bn = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)
        self.activation = Activation('relu')
        self.global_average_pooling = GlobalAvgPool2D()
        self.fc = Dense(c.num_class, name='fully_connected', activation='softmax', use_bias=False, kernel_initializer='he_normal')

    def call(self, inputs, training):
        net = self.conv0(inputs)
        # print('input', inputs.shape)
        # print('conv0', net.shape)

        for block in self.block_collector:
            net = block(net, training)
            # print(block.name, net.shape)
        net = self.bn(net, training)
        net = self.activation(net)

        net = self.global_average_pooling(net)
        # print('global average-pooling', net.shape)
        net = self.fc(net)
        # print('fully connected', net.shape)
        return net

class ResNetv2S1(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(ResNetv2, self).__init__(**kwargs)

        self.conv0 = Conv2D(24, (5, 5), strides=(1, 1), name='conv0', padding='same', use_bias=False, kernel_initializer='he_normal')
        self.block_collector = []
        block_num = 5, 6, 5
        filters_num = 24, 48, 96
        for i in range(1, 4):
            if i == 1:
                self.block_collector.append(BasicBlock(filters_num[i-1], name='conv1_0'))
            else:
                self.block_collector.append(BasicBlock(filters_num[i-1], strides=(2, 2), name='conv{}_0'.format(i)))

            for j in range(1, block_num[i-1]):
                self.block_collector.append(BasicBlock(filters_num[i-1], name='conv{}_{}'.format(i, j)))

        self.bn = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)
        self.activation = Activation('relu')
        self.global_average_pooling = GlobalAvgPool2D()
        self.fc = Dense(c.num_class, name='fully_connected', activation='softmax', use_bias=False, kernel_initializer='he_normal')

    def call(self, inputs, training):
        net = self.conv0(inputs)
        # print('input', inputs.shape)
        # print('conv0', net.shape)

        for block in self.block_collector:
            net = block(net, training)
            # print(block.name, net.shape)
        net = self.bn(net, training)
        net = self.activation(net)

        net = self.global_average_pooling(net)
        # print('global average-pooling', net.shape)
        net = self.fc(net)
        # print('fully connected', net.shape)
        return net

def main():
    model = ResNetv2()
    model.build((None, ) + c.input_shape)

    cnt1 = cnt2 = 0
    for v in model.trainable_variables:
        print(v.name)
        cnt1 += 1
        if 'kernel' in v.name:
            cnt2 += 1
    print(cnt1, cnt2)
    print(model.summary())

if __name__=='__main__':
    main()