# -*- coding:utf-8 -*-
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Resnet model configuration.

References:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition
  arXiv:1512.03385 (2015)

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks
  arXiv:1603.05027 (2016)

  Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy,
  Alan L. Yuille
  DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
  Atrous Convolution, and Fully Connected CRFs
  arXiv:1606.00915 (2016)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


default_config = {
    'image_size': 224,
    'num_classes': 1000,
    }

def conv_bn_layer(inputs, 
                  filters, 
                  filter_size, 
                  stride, 
                  padding='same', 
                  activation=tf.nn.relu,
                  channel_pos='channels_first'):
  """Combination of a convolution layer and a batch normalization layer.

  Args:
    inputs: the input.
    filters: number of filters.
    filter_size: the size of the filter
    stride: the size of the stride
    padding: padding mode, 'same' or 'valid'
    activation: activation function, e.g., tf.nn.relu
    channel_pos: 'channels_first' or 'channels_last'
  """
  kernel_initializer = tf.variance_scaling_initializer()
  if padding != 'same_resnet':
    conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=filter_size,
            strides=stride,
            activation=None,
            padding=padding,
            data_format=channel_pos,
            kernel_initializer=kernel_initializer,
            use_bias=False
            )
  else:
    if stride == 1:
      conv = tf.layers.conv2d(
              inputs=inputs,
              filters=filters,
              kernel_size=filter_size,
              strides=stride,
              padding='same',
              activation=None,
              data_format=channel_pos,
              kernel_initializer=kernel_initializer,
              use_bias=False
              )
    else:
      pad_h_beg = (filter_size - 1) // 2
      pad_h_end = filter_size - 1 - pad_h_beg
      pad_w_beg = (filter_size - 1) // 2
      pad_w_end = filter_size - 1 - pad_w_beg
      padding = [[0, 0], [pad_h_beg, pad_h_end], [pad_w_beg, pad_w_end], [0, 0]]
      if channel_pos == 'channels_first':
        padding = [padding[0], padding[3], padding[1], padding[2]]
      padded_input = tf.pad(inputs, padding)
      conv = tf.layers.conv2d(
              inputs=padded_input,
              filters=filters,
              kernel_size=filter_size,
              strides=stride,
              padding='valid',
              activation=None,
              data_format=channel_pos,
              kernel_initializer=kernel_initializer,
              use_bias=False
              )

    biased = tf.contrib.layers.batch_norm(inputs=conv)
    if activation is not None:
      conv = activation(biased)
    else:
      conv = biased
  return conv

def bottleneck_block_v1(inputs, channels, depth, depth_bottleneck, stride,
        channel_pos):
  """Bottleneck block with identity short-cut for ResNet v1.

  Args:
    inputs: the input images, excluding labels`.
    channels: number of channels for inputs.
    depth: the number of output filters for this bottleneck block.
    depth_bottleneck: the number of bottleneck filters for this block.
    stride: Stride used in the first layer of the bottleneck block.
    channel_pos: 'channels_first' or 'channels_last'
  """
  input_layer = inputs
  in_size = channels

  if depth == in_size:
    if stride == 1:
      shortcut = input_layer
    else:
      shortcut = tf.layers.average_pooling2d(
              inputs=input_layer,
              pool_size=1,
              strides=stride,
              padding='valid',
              data_format=channel_pos
              )
  else:
    shortcut = conv_bn_layer(input_layer, depth, 1, stride, activation=None,
            channel_pos=channel_pos)
  conv1 = conv_bn_layer(input_layer, depth_bottleneck, 1, stride,
          channel_pos=channel_pos)
  conv2 = conv_bn_layer(conv1, depth_bottleneck, 3, 1, padding='same_resnet',
          channel_pos=channel_pos)
  res = conv_bn_layer(conv2, depth, 1, 1, activation=None,
          channel_pos=channel_pos)

  output = tf.nn.relu(shortcut + res)

  return output

def bottleneck_block_v1_5(inputs, channels, depth, depth_bottleneck, stride,
        channel_pos):
  """Bottleneck block with identity short-cut for ResNet v1.5.

  ResNet v1.5 is the informal name for ResNet v1 where stride 2 is used in the
  first 3x3 convolution of each block instead of the first 1x1 convolution.

  First seen at https://github.com/facebook/fb.resnet.torch. Used in the paper
  "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
  (arXiv:1706.02677v2) and by fast.ai to train to accuracy in 45 epochs using
  multiple image sizes.

  Args:
    inputs: the input images, excluding labels`.
    channels: number of channels for inputs.
    depth: the number of output filters for this bottleneck block.
    depth_bottleneck: the number of bottleneck filters for this block.
    stride: Stride used in the first layer of the bottleneck block.
    channel_pos: 'channels_first' or 'channels_last'
  """
  input_layer = inputs
  in_size = channels

  if depth == in_size:
    if stride == 1:
      shortcut = input_layer
    else:
      shortcut = tf.average_pooling2d(
              inputs=input_layer,
              pool_size=1,
              strides=stride,
              padding='valid',
              data_format=channel_pos
              )
  else:
    shortcut = conv_bn_layer(input_layer, depth, 1, stride, activation=None,
            channel_pos=channel_pos)
  conv1 = conv_bn_layer(input_layer, depth_bottleneck, 1, 1, 
          channel_pos=channel_pos)
  conv2 = conv_bn_layer(conv1, depth_bottleneck, 3, stride, 
        padding='same_resnet', channel_pos=channel_pos)
  res = conv_bn_layer(conv2, depth, 1, 1, activation=None,
        channel_pos=channel_pos)
  output = tf.nn.relu(shortcut + res)

  return output

def bottleneck_block_v2(inputs, channels, depth, depth_bottleneck, stride,
        channel_pos):
  """Bottleneck block with identity short-cut for ResNet v2.

  The main difference from v1 is that a batch norm and relu are done at the
  start of the block, instead of the end. This initial batch norm and relu is
  collectively called a pre-activation.

  Args:
    inputs: the input images, excluding labels`.
    channels: number of channels for inputs.
    depth: the number of output filters for this bottleneck block.
    depth_bottleneck: the number of bottleneck filters for this block.
    stride: Stride used in the first layer of the bottleneck block.
    channel_pos: 'channels_first' or 'channels_last'
  """
  input_layer = inputs
  in_size = channels

  preact = tf.contrib.layers.batch_norm(input_layer, activation_fn=tf.nn.relu)
  if depth == in_size:
    if stride == 1:
      shortcut = input_layer
    else:
      shortcut = tf.layers.average_pooling2d(
              inputs=input_layer,
              pool_size=1,
              strides=stride,
              padding='valid',
              data_format=channel_pos
              )
  else:
    shortcut = tf.layers.conv2d(
            inputs=preact,
            filters=depth,
            kernel_size=1,
            strides=stride,
            activation=None,
            use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            padding='same',
            data_format=channel_pos
            )
  conv1 = conv_bn_layer(preact, depth_bottleneck, 1, stride, 
          channel_pos=channel_pos)
  conv2 = conv_bn_layer(conv1, depth_bottleneck, 3, 1, padding='same_resnet', 
          channel_pos=channel_pos)
  res = conv_bn_layer(conv2, depth, 1, 1, activation=None,
          channel_pos=channel_pos)
  output = shortcut + res

  return output

def bottleneck_block(inputs, channels, depth, depth_bottleneck, stride, version,
                     channel_pos):
  """Bottleneck block with identity short-cut.

  Args:
    inputs: the input
    channels: number of channels for inputs
    depth: the number of output filters for this bottleneck block.
    depth_bottleneck: the number of bottleneck filters for this block.
    stride: Stride used in the first layer of the bottleneck block.
    version: version of ResNet to build.
    channel_pos: 'channel_first' or 'channel_last'
  """
  if version == 'v2':
    return bottleneck_block_v2(inputs, channels, depth, depth_bottleneck, 
            stride, channel_pos)
  elif version == 'v1.5':
    return bottleneck_block_v1_5(inputs, channels, depth, depth_bottleneck, 
            stride, channel_pos)
  else:
    return bottleneck_block_v1(inputs, channels, depth, depth_bottleneck, 
            stride, channel_pos)

class ResnetModel(model.Model):
  """Resnet cnn network configuration."""

  def __init__(self, model, layer_counts, params=None):
    self.layer_counts = layer_counts

    super(ResnetModel, self).__init__(
            default_config['image_size'],
            default_config['num_classes'],
            params=params)
    if 'v2' in model:
      self.version = 'v2'
    elif 'v1.5' in model:
      self.version = 'v1.5'
    else:
      self.version = 'v1'

  def build_network(self, inputs, is_training):
    if self.layer_counts is None:
      raise ValueError('Layer counts must be specified.')
    
    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    
    conv1 = conv_bn_layer(inputs, 64, 7, 2, padding='same_resnet',
            channel_pos=self.channel_pos)
    pool1 = tf.layers.max_pooling2d(
            conv1,
            pool_size=3,
            strides=2,
            padding='same',
            data_format=self.channel_pos
            )

    for _ in xrange(self.layer_counts[0]):
      output = bottleneck_block(pool1, 64, 256, 64, 1, 
              self.version, self.channel_pos)
    for i in xrange(self.layer_counts[1]):
      stride = 2 if i == 0 else 1
      output = bottleneck_block(output, 256, 512, 128, 
              stride, self.version, self.channel_pos)
    for i in xrange(self.layer_counts[2]):
      stride = 2 if i == 0 else 1
      output = bottleneck_block(output, 512, 1024, 256, 
              stride, self.version, self.channel_pos)
    for i in xrange(self.layer_counts[3]):
      stride = 2 if i == 0 else 1
      output = bottleneck_block(output, 1024, 2048, 512, 
              stride, self.version, self.channel_pos)
    if self.version == 'v2':
      output = tf.contrib.layers.batch_norm(
              inputs=output,
              activation_fn=tf.nn.relu)

    axes = [1, 2] if self.data_format == 'NHWC' else [2, 3]
    output = tf.reduce_mean(output, axes, keepdims=False)

    stddev = np.sqrt(1.0 / self.num_classes)
    logits = tf.contrib.layers.fully_connected(
            inputs=output,
            num_outputs=self.num_classes,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0)
            )
    return logits

def create_resnet50_model(params):
  return ResnetModel('resnet50', (3, 4, 6, 3), params=params)

def create_resnet50_v1_5_model(params):
  return ResnetModel('resnet50_v1.5', (3, 4, 6, 3), params=params)

def create_resnet50_v2_model(params):
  return ResnetModel('resnet50_v2', (3, 4, 6, 3), params=params)

def create_resnet101_model(params):
  return ResnetModel('resnet101', (3, 4, 23, 3), params=params)

def create_resnet101_v2_model(params):
  return ResnetModel('resnet101_v2', (3, 4, 23, 3), params=params)

def create_resnet152_model(params):
  return ResnetModel('resnet152', (3, 8, 36, 3), params=params)

def create_resnet152_v2_model(params):
  return ResnetModel('resnet152_v2', (3, 8, 36, 3), params=params)
