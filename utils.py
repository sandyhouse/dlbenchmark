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

"""Utilities for tensorflow benchmarks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def get_tensorflow_version():
  version = tf.__version__
  major, minor, _ = version.split('.')
  return (int(major), int(minor))

def get_distribution_strategy(num_gpus, 
                              all_reduce_alg=None,
                              turn_off_distribution_strategy=False):
  """Return a DistributionStrategy for running the model.

  Args:
    num_gpus: Number of GPUs used.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossDeviceOps for available algorithms.
      If None, DistributionStrategy will choose based on device topology.
    turn_off_distribution_strategy: When set to True, do not use any 
      distribution strategy. Note that when it is True, and num_gpus is large 
      than 1, it will raise a ValueError.
  Returns:
    tf.contrib.distribute.DistributionStrategy object.
  Raises:
    ValueError: if turn_off_distribution_strategy is True and num_gpus is
      larger than 1.
  """
  if num_gpus == 0:
    if turn_off_distribution_strategy:
      return None
    else:
      return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    if turn_off_distribution_strategy:
      return None
    else:
      return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  elif turn_off_distribution_strategy:
    raise ValueError("When {} GPUs are specified, "
            "turn_off_distribution_strategy flag cannot be set to "
            "'True'".format(num_gpus))
  else:
    if all_reduce_alg:
      return tf.contrib.distribute.CollectiveAllReduceStrategy(
              num_gpus_per_worker=num_gpus)
    else:
      return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)

