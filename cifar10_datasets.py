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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf  # pylint: disable=g-bad-import-order

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return os.path.join(data_dir, 'train.tfrecords')
  else:
    return os.path.join(data_dir, 'eval.tfrecords')


def parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
  """
  # Dense features in Example proto.
  feature_map = {
      'image': tf.FixedLenFeature([], dtype=tf.string),
      'label': tf.FixedLenFeature([], dtype=tf.int64),
  }

  features = tf.parse_single_example(serialized=example_serialized,
                                     features=feature_map)
  image = tf.decode_raw(features['image'], tf.uint8)
  image.set_shape([32 * 32 * 3])
  image = tf.cast(
          tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0]), tf.float32)
  label = tf.cast(features['label'], dtype=tf.int32)

  return image, label

def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             dtype=tf.float32, datasets_num_private_threads=None,
             num_parallel_batches=1, parse_record_fn=parse_example_proto):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features
    datasets_num_private_threads: Number of private threads for tf.data.
    num_parallel_batches: Number of parallel batches for tf.data.
    parse_record_fn: Function to use for parsing the records.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  dataset = tf.data.TFRecordDataset(filenames).repeat(int(num_epochs))

  dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda value: parse_record_fn(value),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=False))
  dataset = dataset.shuffle(100)
  
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  return dataset
