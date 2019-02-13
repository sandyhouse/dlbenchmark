
"""Imagenet Datasets"""

import os
import tensorflow as tf
from tensorflow.contrib.data.python.ops import threadpool

IMAGE_SIZE = 224
NUM_CHANNELS = 3
NUM_CLASSES = 1001

NUM_TRAIN_FILES = 1024
NUM_EVAL_FILES = 128
SHUFFLE_BUFFER = 10000

NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

def get_files(data_dir, is_training):
  """Return filenames of dataset.

  Args:
    is_training: whether in training phrase
    data_dir: directory for dataset
  Returns:
    the list of filenames
  """
  if is_training:
    return [os.path.join(data_dir, "train-%05d-of-01024" % i) 
            for i in range(NUM_TRAIN_FILES)]
  
  return [os.path.join(data_dir, "validation-%05d-of-00128" % i)
          for i in range(NUM_EVAL_FILES)]

def _parse_example_proto(example):
  """Parse an Example proto containing a training example of an image.
  
  Each Example proto contains the following fields:
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
    example: tf.string containing a serialized Example proto buffer.
  
  Returns:
    image_buffer: tf.string Tensor containing the contents of a JPEG image
    label: tf.int32 Tensor representing the label
  """
  feature_map = {
          'image/encoded': tf.FixedLenFeature([], dtype=tf.string, 
                                              default_value=""),
          'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                                  default_value=-1),
  }

  features = tf.parse_single_example(serialized=example, features=feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  
  return features['image/encoded'], label

def parse_record(raw_record, dtype):
  """Parse a record containing a training example of image.

  The raw record is parsed into a image and label.

  Args:
    raw_record: tf.string Tensor containing a serialized Example proto buffer
    dtype: data type used for images
  
  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label = _parse_example_proto(raw_record)
  image = tf.cast(image, dtype)

  return image, label

def input_fn(is_training, data_dir, batch_size, num_epochs=1, dtype=tf.float32,
             num_parallel_batches=1, parse_record_fn=parse_record):
  """Input function which provides batches for train or eval.
  
  Args:
    is_traing: whether the input is for training.
    data_dir: directory for input dataset.
    batch_size: the number of samples per batch.
    num_epochs: the number of epochs to repeat the dataset.
    dtype: data type used for images.
    datasets_num_private_threads: number of private threads for tf.data.
    num_parallel_batches: number of parallel batches for tf.data.
    parse_record_fn: function to use for parsing the records.

  Returns:
    A dataset object used for iteration.
  """
  files = get_files(data_dir, is_training)
  dataset = tf.data.Dataset.from_tensor_slices(files)
  
  if is_training:
    dataset = dataset.shuffle(buffer_size=NUM_TRAIN_FILES)
  
  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want to
  # increase this number if you have a large number of CPU cores.
  dataset = dataset.apply(tf.data.experimental.parallel_interleave(
      tf.data.TFRecordDataset, cycle_length=10))

  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER)

  dataset = dataset.repeat(num_epochs)

  dataset = dataset.apply(tf.data.experimental.map_and_batch(
      lambda value: parse_record_fn(value, dtype),
      batch_size=batch_size,
      num_parallel_batches=num_parallel_batches,
      drop_remainder=True if is_training else False))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to 
  # background all of the above processing work and keep it out of the critical
  # training path. Setting buffer_size to tf.contrib.data.AUTOTUNE allows 
  # DistributionStrategies to adjust how many batches to fetch based on how many
  # devices are present.
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  return dataset

def get_synth_input_fn(dtype):
  """Returns an input function that returns a dataset with random data."""
  def input_fn(data_dir, is_training, batch_size, dtype, *args, **kwargs):
    """Return dataset filled with random data."""
    # Synthetic input should be within [0, 255].
    inputs = tf.random.truncated_normal(
            [batch_size] + [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS],
            dtype=dtype,
            mean=127,
            stddev=60,
            name='synthetic_inputs')

    labels = tf.random.uniform(
            [batch_size],
            minval=0,
            maxval=NUM_CLASSES - 1,
            dtype=tf.int32,
            name='synthetic_labels')
    data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()
    data = data.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return data

  return input_fn
