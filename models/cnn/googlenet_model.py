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
"""Googlenet model configuration.

References:
  Szegedy, Christian, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
  Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich
  Going deeper with convolutions
  arXiv preprint arXiv:1409.4842 (2014)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from models.cnn import model

default_config = {
    'image_size': 224,
    'num_classes': 1000
    }

def inception_module(inputs, cols, channel_pos):
  input_layer = inputs
  col_layers = []
  for c, col in enumerate(cols):
    col_layers.append([])
    output = None
    for l, layer in enumerate(col):
      ltype, filters = layer[0], layer[1]
      kernel_size = layer[2]
      if len(layer) > 4:
        strides = layer[4]
      else:
        strides=1

      if ltype == 'conv':
        output = tf.layers.conv2d(
                inputs=input_layer if l == 0 else output,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                data_format=channel_pos,
                activation=tf.nn.relu,
                padding='same',
                kernel_initializer=tf.variance_scaling_initializer(),
                bias_initializer=tf.constant_initializer(0)
                )
      elif ltype == 'mpool':
        output = tf.layers.max_pooling2d(
                inputs=input_layer if l == 0 else output,
                pool_size=kernel_size,
                strides=strides,
                data_format=channel_pos,
                padding='same'
                )
      else:
        raise KeyError(
            'Invalid layer type for inception module: \'%s\'' % ltype)
      col_layers[c].append(output)
  catdim = 3 if channel_pos == 'channels_last' else 1
  output = tf.concat([layers[-1] for layers in col_layers], catdim)

  return output

class GooglenetModel(model.Model):
  """GoogLeNet."""

  def __init__(self, params=None):
    super(GooglenetModel, self).__init__(
            default_config['image_size'],
            default_config['num_classes'],
            params=params)

  def build_network(self, inputs, is_training):
    """Builds the forward pass of the model.

    Args:
      inputs: the list of inputs, excluding labels
      is_training: if in the phrase of training.

    Returns:
      The logits of the model.
    """

    def inception_v1(inputs, k, l, m, n, p, q):
      cols = [[('conv', k, 1, 1)], [('conv', l, 1, 1), ('conv', m, 3, 3)],
              [('conv', n, 1, 1), ('conv', p, 5, 5)],
              [('mpool', 3, 3, 1, 1, 'SAME'), ('conv', q, 1, 1)]]
      return inception_module(inputs, cols, self.channel_pos)

    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=64,
            kernel_size=7,
            strides=2,
            padding='same',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.zeros_initializer()
            )
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=3,
            strides=2,
            padding='same',
            data_format=self.channel_pos
            )
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=1,
            strides=1,
            padding='same',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.zeros_initializer()
            )
    conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=192,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.zeros_initializer()
            )
    pool3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=3,
            strides=2,
            padding='same',
            data_format=self.channel_pos
            )
    output1 = inception_v1(pool3, 64, 96, 128, 16, 32, 32)
    output2 = inception_v1(output1, 128, 128, 192, 32, 96, 64)
    pool4 = tf.layers.max_pooling2d(
            inputs=output2,
            pool_size=3,
            strides=2,
            padding='same',
            data_format=self.channel_pos
            )
    output3 = inception_v1(pool4, 192, 96, 208, 16, 48, 64)
    output4 = inception_v1(output3, 160, 112, 224, 24, 64, 64)
    output5 = inception_v1(output4, 128, 128, 256, 24, 64, 64)
    output6 = inception_v1(output5, 112, 144, 288, 32, 64, 64)
    output7 = inception_v1(output6, 256, 160, 320, 32, 128, 128)
    pool8 = tf.layers.max_pooling2d(
            inputs=output7,
            pool_size=3,
            strides=2,
            padding='same',
            data_format=self.channel_pos
            )
    output9 = inception_v1(pool8, 256, 160, 320, 32, 128, 128)
    output10 = inception_v1(output9, 384, 192, 384, 48, 128, 128)
    pool11 = tf.layers.average_pooling2d(
            inputs=output10,
            pool_size=7,
            strides=1,
            padding='valid',
            data_format=self.channel_pos
            )
    output12 = tf.reshape(pool11, [-1, 1024])
    stddev = np.sqrt(1.0 / self.num_classes)
    logits = tf.contrib.layers.fully_connected(
            inputs=output12,
            num_outputs=self.num_classes,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev),
            biases_initializer=tf.constant_initializer(0)
            )
    return logits
