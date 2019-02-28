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
"""Vgg model configuration.

Includes multiple models: vgg11, vgg16, vgg19, corresponding to
  model A, D, and E in Table 1 of [1].

References:
[1]  Simonyan, Karen, Andrew Zisserman
     Very Deep Convolutional Networks for Large-Scale Image Recognition
     arXiv:1409.1556 (2014)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
from models.cnn import model

import tensorflow as tf


default_config = {
    'image_size': 224,
    'num_classes': 1000
    }

def _construct_vgg(inputs, num_conv_layers, channel_pos, is_training):
  """Build vgg architecture from blocks."""
  assert len(num_conv_layers) == 5
  output = inputs
  for _ in xrange(num_conv_layers[0]):
    output = tf.layers.conv2d(
            inputs=output,
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            data_format=channel_pos,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.constant_initializer(0.0)
            )
  output = tf.layers.max_pooling2d(
          inputs=output,
          pool_size=2,
          strides=2,
          padding='valid',
          data_format=channel_pos
          )
  for _ in xrange(num_conv_layers[1]):
    output = tf.layers.conv2d(
            inputs=output,
            filters=128,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            data_format=channel_pos,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.constant_initializer(0.0)
            )   
  output = tf.layers.max_pooling2d(
          inputs=output,
          pool_size=2,
          strides=2,
          padding='valid',
          data_format=channel_pos
          )
  for _ in xrange(num_conv_layers[2]):
    output = tf.layers.conv2d(
            inputs=output,
            filters=256,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            data_format=channel_pos,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.constant_initializer(0.0)
            )
  output = tf.layers.max_pooling2d(
          inputs=output,
          pool_size=2,
          strides=2,
          padding='valid',
          data_format=channel_pos
          )
  for _ in xrange(num_conv_layers[3]):
    output = tf.layers.conv2d(
            inputs=output,
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            data_format=channel_pos,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.constant_initializer(0.0)
            )
  output = tf.layers.max_pooling2d(
          inputs=output,
          pool_size=2,
          strides=2,
          padding='valid',
          data_format=channel_pos
          )
  for _ in xrange(num_conv_layers[4]):
    output = tf.layers.conv2d(
            inputs=output,
            filters=512,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            data_format=channel_pos,
            kernel_initializer=tf.variance_scaling_initializer(),
            bias_initializer=tf.constant_initializer(0.0)
            )
  output = tf.layers.max_pooling2d(
          inputs=output,
          pool_size=2,
          strides=2,
          padding='valid',
          data_format=channel_pos
          )
  output = tf.reshape(output, [-1, 512 * 7 * 7])
  stddev = np.sqrt(1.0 / 4096)
  fc1 = tf.contrib.layers.fully_connected(
          inputs=output,
          num_outputs=4096,
          activation_fn=tf.nn.relu,
          weights_initializer=tf.truncated_normal_initializer(stddev),
          biases_initializer=tf.constant_initializer(0.0)
          )
  drop1 = tf.layers.dropout(
          inputs=fc1,
          rate=0.5,
          training=is_training
          )

  fc2 = tf.contrib.layers.fully_connected(
          inputs=drop1,
          num_outputs=4096,
          activation_fn=tf.nn.relu,
          weights_initializer=tf.truncated_normal_initializer(stddev),
          biases_initializer=tf.constant_initializer(0.0)
          )
  drop2 = tf.layers.dropout(
          inputs=fc2,
          rate=0.5,
          training=is_training
          )

  return drop2

class Vgg11Model(model.Model):

  def __init__(self, params=None):
    super(Vgg11Model, self).__init__(
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
    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    output = _construct_vgg(inputs, [1, 1, 2, 2, 2], self.channel_pos, 
            is_training)
    stdv = np.sqrt(1.0 / self.num_classes)
    logits = tf.contrib.layers.fully_connected(
            inputs=output,
            num_outputs=self.num_classes,
            activation_fn=None,
            weights_initializer=tf.initializers.random_uniform(-stdv, stdv),
            biases_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )
    return logits

class Vgg16Model(model.Model):

  def __init__(self, params=None):
    super(Vgg16Model, self).__init__(
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
    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    output = _construct_vgg(inputs, [2, 2, 3, 3, 3], self.channel_pos, 
            is_training)
    stdv = np.sqrt(1.0 / self.num_classes)
    logits = tf.contrib.layers.fully_connected(
            inputs=output,
            num_outputs=self.num_classes,
            activation_fn=None,
            weights_initializer=tf.initializers.random_uniform(-stdv, stdv),
            biases_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )
    return logits

class Vgg19Model(model.Model):

  def __init__(self, params=None):
    super(Vgg19Model, self).__init__(
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
    if self.data_format == 'NCHW':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    output = _construct_vgg(inputs, [2, 2, 4, 4, 4], self.channel_pos, 
            is_training)
    stdv = np.sqrt(1.0 / self.num_classes)
    logits = tf.contrib.layers.fully_connected(
            inputs=output,
            num_outputs=self.num_classes,
            activation_fn=None,
            weights_initializer=tf.initializers.random_uniform(-stdv, stdv),
            biases_initializer=tf.initializers.random_uniform(-stdv, stdv)
            )
    return logits
