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
"""CNN builder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
import numpy as np

import tensorflow as tf

class ConvNetBuilder(object):
  """Builder of convolutional neural networks."""

  def __init__(self,
               input_op,
               image_depth,
               is_training,
               data_format='NCHW',
               dtype=tf.float32,
               variable_dtype=tf.float32):
    self.top_layer = input_op
    self.top_size = image_depth
    self.phase_train = is_training
    self.data_format = data_format
    self.dtype = dtype
    self.variable_dtype = variable_dtype
    self.counts = defaultdict(lambda: 0)
    self.use_batch_norm = False
    self.batch_norm_config = {}  # 'decay': 0.997, 'scale': True}
    self.channel_pos = ('channels_last'
                        if data_format == 'NHWC' else 'channels_first')

  def get_custom_getter(self):
    """Returns a custom getter that this class's methods must be called under.

    All methods of this class must be called under a variable scope that was
    passed this custom getter. Example:

    ```python
    network = ConvNetBuilder(...)
    with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
      network.conv(...)
      # Call more methods of network here
    ```

    It causes variables to be stored as dtype self.variable_type, then casted 
    to the requested dtype, instead of directly storing the variable as the 
    requested dtype.
    """
    def inner_custom_getter(getter, *args, **kwargs):
      """Custom getter that forces variables to have type self.variable_type."""
      requested_dtype = kwargs['dtype']
      if not (requested_dtype == tf.float32 and
              self.variable_dtype == tf.float16):
        # Only change the variable dtype if doing so does not decrease variable
        # precision.
        kwargs['dtype'] = self.variable_dtype
      var = getter(*args, **kwargs)
      # This if statement is needed to guard the cast, because batch norm
      # assigns directly to the return value of this custom getter. The cast
      # makes the return value not a variable so it cannot be assigned. Batch
      # norm variables are always in fp32 so this if statement is never
      # triggered for them.
      if var.dtype.base_dtype != requested_dtype:
        var = tf.cast(var, requested_dtype)
      return var
    return inner_custom_getter
  
  def get_variable(self, name, shape, dtype, cast_dtype, *args, **kwargs):
    # TODO(reedwm): Currently variables and gradients are transferred to other
    # devices and machines as type `dtype`, not `cast_dtype`. In particular,
    # this means in fp16 mode, variables are transferred as fp32 values, not
    # fp16 values, which uses extra bandwidth.
    var = tf.get_variable(name, shape, dtype, *args, **kwargs)
    return tf.cast(var, cast_dtype)

  def conv(self,
           filters,
           k_height,
           k_width,
           d_height=1,
           d_width=1,
           mode='SAME',
           input_layer=None,
           use_batch_norm=None,
           stddev=None,
           activation='relu',
           bias=0.0,
           kernel_initializer=None):
    """Construct a conv2d layer on top of cnn."""
    if input_layer is None:
      input_layer = self.top_layer
    if stddev is not None and kernel_initializer is None:
      kernel_initializer = tf.truncated_normal_initializer(stddev=stddev)
    if kernel_initializer is None:
      kernel_initializer = tf.variance_scaling_initializer()
    name = 'conv' + str(self.counts['conv'])
    self.counts['conv'] += 1
    with tf.variable_scope(name):
      strides = [1, d_height, d_width, 1]
      if self.data_format == 'NCHW':
        strides = [strides[0], strides[3], strides[1], strides[2]]
      if mode != 'SAME_RESNET':
        conv = tf.layers.conv2d(input_layer, filters,
                                  kernel_size=[k_height, k_width],
                                  strides=[d_height, d_width], padding=mode,
                                  data_format = self.channel_pos,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False)
      else:  # Special padding mode for ResNet models
        if d_height == 1 and d_width == 1:
          conv = tf.layers.conv2d(input_layer, filters,
                                    kernel_size=[k_height, k_width],
                                    strides=[d_height, d_width], padding='SAME',
                                    data_format = self.channel_pos,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=False)
        else:
          pad_h_beg = (k_height - 1) // 2
          pad_h_end = k_height - 1 - pad_h_beg
          pad_w_beg = (k_width - 1) // 2
          pad_w_end = k_width - 1 - pad_w_beg
          padding = [[0, 0], [pad_h_beg, pad_h_end],
                     [pad_w_beg, pad_w_end], [0, 0]]
          if self.data_format == 'NCHW':
            padding = [padding[0], padding[3], padding[1], padding[2]]
          padded_input_layer = tf.pad(input_layer, padding)
          conv = tf.layers.conv2d(padded_input_layer, filters,
                                    kernel_size=[k_height, k_width],
                                    strides=[d_height, d_width],
                                    padding='VALID',
                                    data_format = self.channel_pos,
                                    kernel_initializer=kernel_initializer,
                                    use_bias=False)
      if use_batch_norm is None:
        use_batch_norm = self.use_batch_norm
      if not use_batch_norm:
        if bias is not None:
          biases = self.get_variable('biases', [filters],
                                     self.variable_dtype, self.dtype,
                                     initializer=tf.constant_initializer(bias))
          biased = tf.reshape(
              tf.nn.bias_add(conv, biases, data_format=self.data_format),
              conv.get_shape())
        else:
          biased = conv
      else:
        self.top_layer = conv
        self.top_size = filters
        biased = self.batch_norm(**self.batch_norm_config)
      if activation == 'relu':
        conv1 = tf.nn.relu(biased)
      elif activation == 'linear' or activation is None:
        conv1 = biased
      elif activation == 'tanh':
        conv1 = tf.nn.tanh(biased)
      else:
        raise KeyError('Invalid activation type \'%s\'' % activation)
      self.top_layer = conv1
      self.top_size = filters
      return conv1

  def _pool(self,
            pool_name,
            pool_function,
            k_height,
            k_width,
            d_height,
            d_width,
            mode,
            input_layer,
            num_channels_in):
    """Construct a pooling layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = num_channels_in
    name = pool_name + str(self.counts[pool_name])
    self.counts[pool_name] += 1
    pool = pool_function(
        input_layer, [k_height, k_width], [d_height, d_width],
        padding=mode,
        data_format=self.channel_pos,
        name=name)
    self.top_layer = pool
    return pool

  def mpool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct a max pooling layer."""
    return self._pool('mpool', tf.layers.max_pooling2d, k_height, k_width,
                      d_height, d_width, mode, input_layer, num_channels_in)

  def apool(self,
            k_height,
            k_width,
            d_height=2,
            d_width=2,
            mode='VALID',
            input_layer=None,
            num_channels_in=None):
    """Construct an average pooling layer."""
    return self._pool('apool', tf.layers.average_pooling2d, k_height,
                      k_width, d_height, d_width, mode, input_layer,
                      num_channels_in)

  def reshape(self, shape, input_layer=None):
    if input_layer is None:
      input_layer = self.top_layer
    self.top_layer = tf.reshape(input_layer, shape)
    self.top_size = shape[-1]  # HACK This may not always work
    return self.top_layer

  def affine(self,
             num_out_channels,
             input_layer=None,
             num_channels_in=None,
             bias=0.0,
             stddev=None,
             activation='relu'):
    if input_layer is None:
      input_layer = self.top_layer
    if num_channels_in is None:
      num_channels_in = self.top_size
    name = 'affine' + str(self.counts['affine'])
    self.counts['affine'] += 1
    with tf.variable_scope(name) as scope:
      init_factor = 2. if activation == 'relu' else 1.
      if activation == 'relu':
        activation = tf.nn.relu
      elif activation == 'linear' or activation == None:
        activation = None
      elif activation == 'softmax':
        activation = tf.nn.softmax
      else:
        raise ValueError("Not supported activation: %s\n", activation)
      stddev = stddev or np.sqrt(init_factor / num_channels_in)
      affine1 = tf.contrib.layers.fully_connected(input_layer, num_out_channels, 
            activation, 
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(bias),
            scope=scope)
      self.top_layer = affine1
      self.top_size = num_out_channels
      return affine1

  def inception_module(self, name, cols, input_layer=None, in_size=None):
    if input_layer is None:
      input_layer = self.top_layer
    if in_size is None:
      in_size = self.top_size
    name += str(self.counts[name])
    self.counts[name] += 1
    with tf.variable_scope(name):
      col_layers = []
      col_layer_sizes = []
      for c, col in enumerate(cols):
        col_layers.append([])
        col_layer_sizes.append([])
        for l, layer in enumerate(col):
          ltype, args = layer[0], layer[1:]
          kwargs = {
              'input_layer': input_layer,
          } if l == 0 else {}
          if ltype == 'conv':
            self.conv(*args, **kwargs)
          elif ltype == 'mpool':
            self.mpool(*args, **kwargs)
          elif ltype == 'apool':
            self.apool(*args, **kwargs)
          elif ltype == 'share':  # Share matching layer from previous column
            self.top_layer = col_layers[c - 1][l]
            self.top_size = col_layer_sizes[c - 1][l]
          else:
            raise KeyError(
                'Invalid layer type for inception module: \'%s\'' % ltype)
          col_layers[c].append(self.top_layer)
          col_layer_sizes[c].append(self.top_size)
      catdim = 3 if self.data_format == 'NHWC' else 1
      self.top_layer = tf.concat([layers[-1] for layers in col_layers], catdim)
      self.top_size = sum([sizes[-1] for sizes in col_layer_sizes])
      return self.top_layer

  def spatial_mean(self, keep_dims=False):
    name = 'spatial_mean' + str(self.counts['spatial_mean'])
    self.counts['spatial_mean'] += 1
    axes = [1, 2] if self.data_format == 'NHWC' else [2, 3]
    self.top_layer = tf.reduce_mean(
        self.top_layer, axes, keepdims=keep_dims, name=name)
    return self.top_layer

  def dropout(self, keep_prob=0.5, input_layer=None):
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'dropout' + str(self.counts['dropout'])
    with tf.variable_scope(name):
      if not self.phase_train:
        keep_prob = 1.0
      dropout = tf.layers.dropout(input_layer, 1. - keep_prob,
                                    training=self.phase_train)
      self.top_layer = dropout
      return dropout

  def batch_norm(self, input_layer=None, decay=0.999, scale=False,
                 epsilon=0.001, center=True):
    """Adds a Batch Normalization layer."""
    if input_layer is None:
      input_layer = self.top_layer
    else:
      self.top_size = None
    name = 'batchnorm' + str(self.counts['batchnorm'])
    self.counts['batchnorm'] += 1

    with tf.variable_scope(name) as scope:
      bn = tf.contrib.layers.batch_norm(
          input_layer,
          decay=decay,
          scale=scale,
          epsilon=epsilon,
          is_training=self.phase_train,
          data_format=self.data_format,
          scope=scope,
          center=center)
    self.top_layer = bn
    self.top_size = bn.shape[3] if self.data_format == 'NHWC' else bn.shape[1]
    self.top_size = int(self.top_size)
    return bn

  def lrn(self, depth_radius=5, bias=1, alpha=1, beta=0.5):
    """Adds a local response normalization layer."""
    name = 'lrn' + str(self.counts['lrn'])
    self.counts['lrn'] += 1
    self.top_layer = tf.nn.lrn(
        self.top_layer, depth_radius, bias, alpha, beta, name=name)
    return self.top_layer
