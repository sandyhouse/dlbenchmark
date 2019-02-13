# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
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
"""Base model configuration for CNN benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import convnet_builder

class Model(object):
  """Base class for convolutional neural networks."""

  def __init__(self,
               img_height,
               batch_size,
               learning_rate,
               nclass=1001,
               params=None):
    """
    Args:
      img_height: height of the input images. We assume the height of images
        and the width of the images are the same.
      batch_size: batch size per step
      learning_rate: initial learning rate
      nclass: number of classes for dataset, e.g., 10 for Cifar10
      params: other parameters
    """
    self.image_size = img_height
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.depth = 3 # number of channels for images
    self.data_format = params.data_format if params else 'NCHW'
    self.num_classes = nclass

    # fp16_vars specifies whether to create the variables in float16.
    if params:
      self.fp16_vars = params.fp16_vars
      self.data_type = tf.float16 if params.use_fp16 else tf.float32
    else:
      self.fp16_vars = False
      self.data_type = tf.float32

  def get_learning_rate(self, global_step, batch_size):
    del global_step
    del batch_size
    return self.learning_rate

  def get_input_shapes(self, subset):
    """Returns the list of expected shapes of all the inputs to this model."""
    del subset  # Same shapes for both 'train' and 'validation' subsets.
    # Each input is of shape [batch_size, height, width, depth]
    # Each label is of shape [batch_size]
    return [[self.batch_size, self.image_size, self.image_size, self.depth],
            [self.batch_size]]

  def get_input_data_types(self, subset):
    """Returns the list of data types of all the inputs to this model."""
    del subset
    return [self.data_type, tf.int32] # for both image and label

  def get_synthetic_inputs(self, input_name, nclass):
    """Returns the ops to generate synthetic inputs."""
    # Synthetic input should be within [0, 255].
    image_shape, label_shape = self.get_input_shapes('train')
    inputs = tf.truncated_normal(
        image_shape,
        dtype=self.data_type,
        mean=127,
        stddev=60,
        name='synthetic_inputs')
    inputs = tf.contrib.framework.local_variable(inputs, name=input_name)
    labels = tf.random_uniform(
        label_shape,
        minval=0,
        maxval=nclass - 1,
        dtype=tf.int32,
        name='synthetic_labels')

    return (inputs, labels)

  def add_inference(self, cnn):
    """Adds the core layers of the CNN's forward pass.
    This should build the forward pass layers, except for the initial transpose
    of the images and the final Dense layer producing the logits. The layers
    should be build with the ConvNetBuilder `cnn`, so that when this function
    returns, `cnn.top_layer` and `cnn.top_size` refer to the last layer and the
    number of units of the layer layer, respectively.
    Args:
      cnn: A ConvNetBuilder to build the forward pass layers with.
    """
    del cnn
    raise NotImplementedError('Must be implemented in derived classes')

  def build_network(self, inputs, phase_train):
    """Builds the forward pass of the model.

    Args:
      inputs: The list of inputs, not including labels
      phase_train: True during training. False during evaluation.

    Returns:
      The logits of the model.
    """
    images = inputs
    if self.data_format == 'NCHW':
      images = tf.transpose(images, [0, 3, 1, 2])
    var_type = tf.float32
    if self.data_type == tf.float16 and self.fp16_vars:
      var_type = tf.float16
    network = convnet_builder.ConvNetBuilder(
        images, self.depth, phase_train, self.data_format,
        self.data_type, var_type)
    with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
      self.add_inference(network)
      # Add the final fully-connected class layer
      logits = (
          network.affine(self.num_classes, activation='linear'))
          #network.affine(self.num_classes, activation='softmax'))
    if self.data_type == tf.float16:
      logits = tf.cast(logits, tf.float32)
    return logits

  def loss_function(self, inputs, logits):
    """Returns the op to measure the loss of the model.

    Args:
      inputs: the input list of the model.
      logits: the logits of the model.

    Returns:
      The loss tensor of the model.
    """
    _, labels = inputs
    with tf.name_scope('xentropy'):
      cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          logits=logits, labels=labels)
      loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

  def accuracy_function(self, inputs, logits):
    """Returns the ops to measure the accuracy of the model."""
    _, labels = inputs
    top_1_op = tf.reduce_sum(
        tf.cast(tf.nn.in_top_k(logits, labels, 1), self.data_type))
    top_5_op = tf.reduce_sum(
        tf.cast(tf.nn.in_top_k(logits, labels, 5), self.data_type))
    return {'top_1_accuracy': top_1_op, 'top_5_accuracy': top_5_op}

  def postprocess(self, results):
    """Postprocess results returned from model in Python."""
    return results

  def reached_target(self):
    """Define custom methods to stop training when model's target is reached."""
    return False
