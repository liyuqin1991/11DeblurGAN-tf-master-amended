import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import math

FUNCTION
Conv(name, x, filter_size, in_filters, out_filters, strides, padding)
WITH
variable_scope(name)
DO
n = filter_size * filter_size * out_filters
kernel = get_variable('filter', [filter_size, filter_size, in_filters, out_filters], float32,
                      initializer=random_normal_initializer(stddev=0.01))
bias = get_variable('bias', [out_filters], float32, initializer=zeros_initializer())

RETURN
nn.conv2d(x, kernel, [1, strides, strides, 1], padding=padding) + bias
END
FUNCTION

FUNCTION
Conv_transpose(name, x, filter_size, in_filters, out_filters, fraction, padding)
WITH
variable_scope(name)
DO
n = filter_size * filter_size * out_filters
kernel = get_variable('filter', [filter_size, filter_size, out_filters, in_filters], float32,
                      initializer=random_normal_initializer(stddev=sqrt(2.0 / n)))
size = shape(x)
output_shape = stack([size[0], size[1] * fraction, size[2] * fraction, out_filters])
x = nn.conv2d_transpose(x, kernel, output_shape, [1, fraction, fraction, 1], padding)

RETURN
x
END
FUNCTION

FUNCTION
instance_norm(name, x, dim, affine, BN_decay, BN_epsilon)
mean, variance = nn.moments(x, axes=[1, 2])
x = (x - mean) / ((variance + BN_epsilon) ** 0.5)

IF
affine
THEN
beta = get_variable(name + "beta", shape=dim, dtype=float32, initializer=constant_initializer(0.0, float32))
gamma = get_variable(name + "gamma", dim, float32, initializer=constant_initializer(1.0, float32))
x = gamma * x + beta

RETURN
x
END
FUNCTION
FUNCTION
att_conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):

WITH
variable_scope(scope):
IF
pad_type == 'zero':
x = pad_tensor(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
IF
pad_type == 'reflect':
x = pad_tensor(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

IF
sn:
w = get_variable("kernel", shape=[kernel, kernel, get_shape(x)[-1], channels],
                 initializer=weight_init, regularizer=weight_regularizer)
x = conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding='VALID')
IF
use_bias:
bias = get_variable("bias", [channels], initializer=constant_initializer(0.0))
x = bias_add(x, bias)

ELSE:
x = conv2d(inputs=x, filters=channels, kernel_size=kernel, kernel_initializer=weight_init,
           kernel_regularizer=weight_regularizer, strides=stride, use_bias=use_bias)

return x

FUNCTION
hw_flatten(x):
return reshape(x, shape=[1, -1, get_shape(x)[-1]])