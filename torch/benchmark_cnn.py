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
"""TensorFlow benchmarks for Convolutional Neural Networks (CNNs)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

import torch

def get_optimizer(model, params, learning_rate):
  """Returns the optimizer that should be used based on params."""
  if params.optimizer == 'momentum':
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=params.momentum, nesterov=True)
  elif params.optimizer == 'sgd':
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
  elif params.optimizer == 'adam':
    opt = torch.optim.Adam(model.parameters(),
                           lr=learning_rate, 
                           betas=(params.adam_beta1, params.adam_beta2), 
                           eps=params.adam_epsilon)
  else:
    raise ValueError('Optimizer "{}" was not recognized'.
                     format(params.optimizer))
  return opt

class BenchmarkCNN(object):
  """Class for benchmarking a cnn network."""

  def __init__(self, params):
    """Initialize BenchmarkCNN.

    Args:
      params: Params tuple, created by make_params_from_flags.
    Raises:
      ValueError: Unsupported params settings.
    """
    self.params = params

    self.model = self.params.model
    self.batch_size_per_device = self.params.batch_size
    self.batch_size = self.params.batch_size

    self.num_gpus = self.params.num_gpus
    if self.num_gpus:
      self.batch_size = self.batch_size_per_device * self.params.num_gpus

    self.do_train = self.params.do_train
    self.do_eval = self.params.do_eval

    self.num_epochs = self.params.num_epochs

    self.use_synthetic_data = False if self.params.data_dir else True
    self.data_dir = self.params.data_dir

    if self.use_fp16 and self.fp16_vars:
      self.data_type = torch.float16
    else:
      self.data_type = torch.float32

    self.print_info()
    
  def print_info(self):
    """Print basic information."""
    dataset_name = "ImageNet-synthetic" if self.use_synthetic_data else (
                   "ImageNet")
    mode = ''
    if self.do_train:
      mode += 'train '
    if self.do_eval:
      mode += 'eval '

    print()
    print('Model:       %s' % self.model)
    print('Dataset:     %s' % dataset_name)
    print('Mode:        %s' % mode)
    print('Batch size:  %s global (per machine)' % (
           self.batch_size))
    print('             %s per device' % (self.batch_size_per_device))
    print('Num GPUs:    %d per worker' % (self.num_gpus))
    print('Num epochs:  %d' % self.num_epochs)
    print('Data format: %s' % self.data_format)
    print('Optimizer:   %s' % self.optimizer)
    print('AllReduce:   %s' % self.all_reduce_spec)
    print('=' * 30)

  def run(self):
    """Run the benchmark task assigned to this process."""
    pass
