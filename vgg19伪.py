#引入所需要的模块
import os
import tensorflow as tf

import numpy as np
import time
import inspect

'''

I modified the source code from https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
to use generally VGG19 pre-trained model

1. I removed input size limitation
2. I separated relu_layer and conv_layer
3. I removed the fully connected layer because these layer are no need for image style transfer

'''


VGG_MEAN = [100, 100, 100]

VGG_MEAN = [100, 100, 100]


class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path 为空:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            输出
            vgg19_npy_path
        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        输出
        "npy file loaded"

    def build(self, rgb):
        开始计时
        输出
        "build model started"
        将
        rgb
        矩阵值缩放为[-1, 1]
        rgb_scaled = ((rgb + 1) * 255.0) / 2.0


# 将 RGB 转为 BGR
red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
bgr = tf.concat(axis=3, values=[
    blue - VGG_MEAN[0],
    green - VGG_MEAN[1],
    red - VGG_MEAN[2],
])

# 构建卷积神经网络 VGG16 的前 5 层
self.conv1_1 = self.conv_layer(bgr, "conv1_1")
self.relu1_1 = self.relu_layer(self.conv1_1, "relu1_1")
self.conv1_2 = self.conv_layer(self.relu1_1, "conv1_2")
self.relu1_2 = self.relu_layer(self.conv1_2, "relu1_2")
self.pool1 = self.max_pool(self.relu1_2, 'pool1')

self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
self.relu2_1 = self.relu_layer(self.conv2_1, "relu2_1")
self.conv2_2 = self.conv_layer(self.relu2_1, "conv2_2")
self.relu2_2 = self.relu_layer(self.conv2_2, "relu2_2")
self.pool2 = self.max_pool(self.relu2_2, 'pool2')

self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
self.relu3_1 = self.relu_layer(self.conv3_1, "relu3_1")
self.conv3_2 = self.conv_layer(self.relu3_1, "conv3_2")
self.relu3_2 = self.relu_layer(self.conv3_2, "relu3_2")
self.conv3_3 = self.conv_layer(self.relu3_2, "conv3_3")
self.relu3_3 = self.relu_layer(self.conv3_3, "relu3_3")
self.conv3_4 = self.conv_layer(self.relu3_3, "conv3_4")
self.relu3_4 = self.relu_layer(self.conv3_4, "relu3_4")
self.pool3 = self.max_pool(self.relu3_4, 'pool3')

self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
self.relu4_1 = self.relu_layer(self.conv4_1, "relu4_1")
self.conv4_2 = self.conv_layer(self.relu4_1, "conv4_2")
self.relu4_2 = self.relu_layer(self.conv4_2, "relu4_2")
self.conv4_3 = self.conv_layer(self.relu4_2, "conv4_3")
self.relu4_3 = self.relu_layer(self.conv4_3, "relu4_3")
self.conv4_4 = self.conv_layer(self.relu4_3, "conv4_4")
self.relu4_4 = self.relu_layer(self.conv4_4, "relu4_4")
self.pool4 = self.max_pool(self.relu4_4, 'pool4')

self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
self.relu5_1 = self.relu_layer(self.conv5_1, "relu5_1")
self.conv5_2 = self.conv_layer(self.relu5_1, "conv5_2")
self.relu5_2 = self.relu_layer(self.conv5_2, "relu5_2")
self.conv5_3 = self.conv_layer(self.relu5_2, "conv5_3")
self.relu5_3 = self.relu_layer(self.conv5_3, "relu5_3")
self.conv5_4 = self.conv_layer(self.relu5_3, "conv5_4")
self.relu5_4 = self.relu_layer(self.conv5_4, "relu5_4")
self.pool5 = self.max_pool(self.conv5_4, 'pool5')
# 定义全连接层
self.fc6 = self.fc_layer(self.pool5, "fc6")
assert self.fc6.get_shape().as_list()[1:] == [4096]
self.relu6 = self.relu_layer(self.fc6, "relu6")

self.fc7 = self.fc_layer(self.relu6, "fc7")
self.relu7 = self.relu_layer(self.fc7, "relu7")

self.fc8 = self.fc_layer(self.relu7, "fc8")

# 定义输出层
self.prob = tf.nn.softmax(self.fc8, name="prob")

# 输出模型建立完成的时间
self.data_dict = None
print(("build model finished: %ds" % (time.time() - start_time)))


# 定义平均池化层
def avg_pool(self, bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


# 定义最大池化层
def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


# 定义一个ReLU层
def relu_layer(self, bottom, name):


# 应用ReLU激活函数
return tf.nn.relu(bottom, name=name)


# 定义一个卷积层
def conv_layer(self, bottom, name):


# 设置变量域
with tf.variable_scope(name):
# 获取卷积核
filt = self.get_conv_filter(name)
php
Copy
code
# 进行卷积操作
conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
# 获取偏置项
conv_biases = self.get_bias(name)
# 添加偏置项
bias = tf.nn.bias_add(conv, conv_biases)
# 应用ReLU激活函数
relu = tf.nn.relu(bias)
bash
Copy
code
# 返回输出结果
return bias


# 定义一个全连接层
def fc_layer(self, bottom, name):


# 设置变量域
with tf.variable_scope(name):
# 获取输入形状
shape = bottom.get_shape().as_list()
# 计算输入的维度
dim = 1
for d in shape[1:]:
    dim *= d
# 将输入展开为二维张量
x = tf.reshape(bottom, [-1, dim])

bash
Copy
code
# 获取权重和偏置项
weights = self.get_fc_weight(name)
biases = self.get_bias(name)

# 全连接层计算
fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

# 返回输出结果
return fc
获取卷积核


def get_conv_filter(self, name):
    return tf.constant(self.data_dict[name][0], name="filter")


获取偏置项


def get_bias(self, name):
    return tf.constant(self.data_dict[name][1], name="biases")


获取全连接层权重


def get_fc_weight(self, name):
    return tf.constant(self.data_dict[name][0], name="weights")
