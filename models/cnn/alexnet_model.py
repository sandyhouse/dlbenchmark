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
"""Alexnet model configuration.

References:
  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton
  ImageNet Classification with Deep Convolutional Neural Networks
  Advances in Neural Information Processing Systems. 2012
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from models.cnn import model

default_config = {
    'image_size': 224,
    'num_classes': 1000
    }

class AlexnetModel(model.Model):
  """Alexnet model for ImageNet."""

  def __init__(self, params=None):
    super(AlexnetModel, self).__init__(
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
    inputs = tf.image.resize_image_with_crop_or_pad(inputs, 227, 227)
    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    stdv = 1.0 / math.sqrt(inputs.shape.as_list()[1] * 11 * 11)
    conv1 = tf.layers.conv2d(
            inputs=inputs, 
            filters=64,
            kernel_size=11,
            strides=4,
            padding='valid',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.random_uniform(-stdv, stdv),
            bias_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=3,
            strides=2,
            padding='valid',
            data_format=self.channel_pos
            )

    stdv = 1.0 / math.sqrt(pool1.shape.as_list()[1] * 5 * 5)
    conv2 = tf.layers.conv2d(
            inputs=pool1, 
            filters=192,
            kernel_size=5,
            strides=1,
            padding='same',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.random_uniform(-stdv, stdv),
            bias_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=3,
            strides=2,
            padding='valid',
            data_format=self.channel_pos
            )

    stdv = 1.0 / math.sqrt(pool2.shape.as_list()[1] * 3 * 3)
    conv3 = tf.layers.conv2d(
            inputs=pool2, 
            filters=384,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.random_uniform(-stdv, stdv),
            bias_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )

    stdv = 1.0 / math.sqrt(conv3.shape.as_list()[1] * 3 * 3)
    conv4 = tf.layers.conv2d(
            inputs=conv3, 
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.random_uniform(-stdv, stdv),
            bias_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )

    stdv = 1.0 / math.sqrt(conv4.shape.as_list()[1] * 3 * 3)
    conv5 = tf.layers.conv2d(
            inputs=conv4, 
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.random_uniform(-stdv, stdv),
            bias_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )
    pool5 = tf.layers.max_pooling2d(
            inputs=conv5,
            pool_size=3,
            strides=2,
            padding='valid',
            data_format=self.channel_pos
            )

    reshape = tf.reshape(pool5, [-1, 256 * 6 * 6])

    drop6 = tf.layers.dropout(
            inputs=reshape,
            rate=0.5,
            training=is_training
            )

    stdv = 1.0 / math.sqrt(self.num_classes * 1.0)
    fc6 = tf.contrib.layers.fully_connected(
            inputs=drop6,
            num_outputs=4096,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.initializers.random_uniform(-stdv, stdv),
            biases_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )

    drop7 = tf.layers.dropout(
            inputs=fc6,
            rate=0.5,
            training=is_training
            )

    stdv = 1.0 / math.sqrt(drop7.shape.as_list()[1] * 1.0)
    fc7 = tf.contrib.layers.fully_connected(
            inputs=drop7,
            num_outputs=4096,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.initializers.random_uniform(-stdv, stdv),
            biases_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )

    stdv = 1.0 / math.sqrt(fc7.shape.as_list()[1] * 1.0)
    logits = tf.contrib.layers.fully_connected(
            inputs=fc7,
            num_outputs=self.num_classes,
            activation_fn=None,
            weights_initializer=tf.initializers.random_uniform(-stdv, stdv),
            biases_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )

    return logits


class AlexnetCifar10Model(model.Model):
  """Alexnet model for ImageNet."""

  def __init__(self, params=None):
    super(AlexnetCifar10Model, self).__init__(
            32,
            10,
            params=params)

  def build_network(self, inputs, is_training):
    """Builds the forward pass of the model.

    Args:
      inputs: the list of inputs, excluding labels
      is_training: if in the phrase of training.

    Returns:
      The logits of the model.
    """
    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])

    stdv = 1.0 / math.sqrt(inputs.shape.as_list()[1] * 11 * 11)
    conv1 = tf.layers.conv2d(
            inputs=inputs, 
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.random_uniform(-stdv, stdv),
            bias_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )
    pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=3,
            strides=2,
            padding='same',
            data_format=self.channel_pos
            )

    stdv = 1.0 / math.sqrt(pool1.shape.as_list()[1] * 5 * 5)
    conv2 = tf.layers.conv2d(
            inputs=pool1, 
            filters=64,
            kernel_size=5,
            strides=1,
            padding='same',
            data_format=self.channel_pos,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.random_uniform(-stdv, stdv),
            bias_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )
    pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=3,
            strides=2,
            padding='same',
            data_format=self.channel_pos
            )

    shape = pool2.get_shape().as_list()
    flat_dim = shape[1] * shape[2] * shape[3]
    reshape = tf.reshape(pool2, [-1, flat_dim])

    stdv = 1.0 / math.sqrt(self.num_classes * 1.0)
    fc3 = tf.contrib.layers.fully_connected(
            inputs=reshape,
            num_outputs=384,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.initializers.random_uniform(-stdv, stdv),
            biases_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )

    fc4 = tf.contrib.layers.fully_connected(
            inputs=fc3,
            num_outputs=192,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.initializers.random_uniform(-stdv, stdv),
            biases_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )

    stdv = 1.0 / math.sqrt(fc4.shape.as_list()[1] * 1.0)
    logits = tf.contrib.layers.fully_connected(
            inputs=fc4,
            num_outputs=self.num_classes,
            activation_fn=None,
            weights_initializer=tf.initializers.random_uniform(-stdv, stdv),
            biases_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )

    return logits
