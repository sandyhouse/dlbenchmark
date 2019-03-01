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

import imagenet_preprocessing

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1000

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.io.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
      'image/class/label': tf.io.FixedLenFeature([], dtype=tf.int64,
                                                 default_value=-1),
      'image/class/text': tf.io.FixedLenFeature([], dtype=tf.string,
                                                default_value=''),
  }
  sparse_float32 = tf.io.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  #feature_map.update(
  #    {k: sparse_float32 for k in ['image/object/bbox/xmin',
  #                                 'image/object/bbox/ymin',
  #                                 'image/object/bbox/xmax',
  #                                 'image/object/bbox/ymax']})

  features = tf.io.parse_single_example(serialized=example_serialized,
                                        features=feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  #xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  #ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  #xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  #ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  ## Note that we impose an ordering of (y, x) just to make life difficult.
  #bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  ## Force the variable number of bounding boxes into the shape
  ## [1, num_boxes, coords].
  #bbox = tf.expand_dims(bbox, 0)
  #bbox = tf.transpose(a=bbox, perm=[0, 2, 1])

  #return features['image/encoded'], label, bbox
  return features['image/encoded'], label


def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  #image_buffer, label, bbox = _parse_example_proto(raw_record)
  image_buffer, label = _parse_example_proto(raw_record)

  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      bbox=None,
      output_height=DEFAULT_IMAGE_SIZE,
      output_width=DEFAULT_IMAGE_SIZE,
      num_channels=NUM_CHANNELS,
      is_training=is_training)
  image = tf.cast(image, dtype)

  return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             dtype=tf.float32, datasets_num_private_threads=None,
             num_parallel_batches=1, parse_record_fn=parse_record):
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
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  dataset = dataset.apply(tf.data.experimental.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=10))

  if datasets_num_private_threads:
    options = tf.data.Options()
    options.experimental_threading = tf.data.experimental.ThreadingOptions()
    options.experimental_threading.private_threadpool_size = (
            datasets_num_private_threads)
    dataset = dataset.with_options(options)
    tf.logging.info('datasets_num_private_threads: %s', 
                    datasets_num_private_threads)

  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = dataset.repeat(int(num_epochs))

  dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda value: parse_record_fn(value, is_training, dtype),
            batch_size=batch_size,
            num_parallel_batches=num_parallel_batches,
            drop_remainder=False))
  
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  return dataset

def get_synth_input_fn(dtype):
  def input_fn(is_training, data_dir, batch_size, *args, **kwargs):
    inputs = tf.random.truncated_normal(
        [batch_size] + [DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, NUM_CHANNELS],
        dtype=dtype,
        mean=127,
        stddev=60,
        name='synthetic_inputs')

    labels = tf.random.uniform(
        [batch_size],
        minval=0,
        maxval=NUM_CLASSES-1,
        dtype=tf.int32,
        name='synthetic_labels')

    num_images = NUM_IMAGES['train'] if is_training else (
            NUM_IMAGES['validation'])

    num_images = num_images * kwargs['num_epochs']

    #repeated_num = (num_images + batch_size - 1) // batch_size

    data = tf.data.Dataset.from_tensors((inputs, labels)).repeat(8000)
    data = data.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return data

  return input_fn
