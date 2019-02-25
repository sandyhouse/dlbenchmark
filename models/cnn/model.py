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
"""Base model class for convolutional neural network models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Model(object):
  """Base class for convolutional neural networks."""

  def __init__(self,
               image_size,
               num_classes,
               params):
    """
    Args:
      imgage_size: height/width of the input images. We assume the height and 
        the width of images are of the same.
      batch_size: batch size per iteration
      learning_rate: initial learning rate
      nclass: number of classes for dataset, e.g., 10 for Cifar10, 100 for 
        Cifar100 and 1000 for ImageNet.
      params: other parameters, an object of argparse.parser()

    Returns:
      None
    """
    self.image_size = image_size
    self.batch_size = params.batch_size
    self.learning_rate = params.init_learning_rate
    self.depth = 3 # number of channels for images, always be 3.
    self.data_format = params.data_format
    self.num_classes = num_classes

    self.data_type = tf.float16 if params.use_fp16 else tf.float32

    if self.data_format == 'NCHW':
      self.channel_pos = 'channels_first'
    else:
      self.channel_pos = 'channels_last'

  def build_network(self, inputs, is_training):
    """Builds the forward pass of the model.

    Args:
      inputs: the list of inputs, excluding labels
      is_training: if in the phrase of training.

    Returns:
      The logits of the model.
    """
    raise NotImplementedError("This method must be implemented in subclasses.")
