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

import tensorflow as tf

from models.cnn import alexnet_model as alexnet
from models.cnn import resnet_model as resnet
from models.cnn import googlenet_model as googlenet
from models.cnn import vgg_model as vgg
from models.cnn import trivial_model as trivial
import imagenet_datasets as datasets
import utils

def get_optimizer(params, learning_rate):
  """Returns the optimizer that should be used based on params."""
  if params.optimizer == 'momentum':
    opt = tf.train.MomentumOptimizer(
        learning_rate, params.momentum, use_nesterov=True)
  elif params.optimizer == 'sgd':
    opt = tf.train.GradientDescentOptimizer(learning_rate)
  elif params.optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(
        learning_rate,
        params.rmsprop_decay,
        momentum=params.rmsprop_momentum,
        epsilon=params.rmsprop_epsilon)
  elif params.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate, params.adam_beta1,
                                 params.adam_beta2, params.adam_epsilon)
  else:
    raise ValueError('Optimizer "{}" was not recognized'.
                     format(params.optimizer))
  return opt

def get_learning_rate(params, global_step, num_examples_per_epoch, batch_size):
  """Get a learning rate that decays step-wise."""
  with tf.name_scope("learning_rate"):
    num_batches_per_epoch = num_examples_per_epoch // batch_size

    if params.init_learning_rate is not None:
      learning_rate = params.init_learning_rate
      if(params.num_epochs_per_decay > 0 and
         params.learning_rate_decay_factor > 0):
        decay_steps = num_batches_per_epoch * params.num_epochs_per_decay

        learning_rate = tf.train.exponential_decay(
                params.init_learning_rate,
                global_step,
                decay_steps,
                params.learning_rate_decay_factor,
                staircase=True)

        if params.minimum_learning_rate != 0.:
          learning_rate = tf.maximum(learning_rate,
                                     params.minimum_learning_rate)
          return learning_rate
      else:
        return learning_rate
    else:
      raise ValueError("`--init_learning_rate` must be specified.")

def model_fn(features, labels, mode, params):
  """Define how to train, evaluate and predict."""
  tf.summary.image('images', features, max_outputs=6)
  with tf.variable_scope("model"):
    # Create model and get output logits.
    if params.model == "alexnet":
      model = alexnet.AlexnetModel(params)
    elif params.model == "resnet50":
      model = resnet.create_resnet50_model(params)
    elif params.model == "googlenet":
      model = googlenet.GooglenetModel(params)
    elif params.model == "vgg16":
      model = vgg.Vgg16Model(params)
    else:
      raise ValueError("model `{}` not supported.".format(params.model))

    logits = model.build_network(features, mode==tf.estimator.ModeKeys.TRAIN)
    logits = tf.cast(logits, tf.float32)

    predictions = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
    }

    # When in prediction mode, the labels/targets is None. The model output
    # is the prediction.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(
              mode=mode,
              predictions=predictions,
              export_outputs={
                  'predict': tf.estimator.export.PredictOutput(predictions)
              })

    # Calculate loss, including softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)

    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    loss = cross_entropy
    
    if mode == tf.estimator.ModeKeys.EVAL:
      train_op = None

    elif mode == tf.estimator.ModeKeys.TRAIN:
      # FIXME: fix the params.batch_size based on num_gpus
      num_examples_per_epoch = datasets.NUM_IMAGES['train'] // params.batch_size
      global_step = tf.train.get_or_create_global_step()
      learning_rate = get_learning_rate(
              params,
              global_step,
              num_examples_per_epoch,
              params.batch_size)

      tf.identity(learning_rate, name='learning_rate')
      tf.summary.scalar('learning_rate', learning_rate)

      optimizer = get_optimizer(params, learning_rate)

      #minimize_op = optimizer.minimize(loss, global_step=global_step)
      train_op = optimizer.minimize(loss, global_step=global_step)

      #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      #train_op = tf.group(minimize_op, update_ops)

    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.metrics.mean(
            tf.nn.in_top_k(predictions=logits, targets=labels, k=5,
                name='top_5_op'))
    metrics = {'accuracy': accuracy,
               'accuracy_top_5': accuracy_top_5}

    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.summary.scalar('train_accuracy', accuracy[1])
    tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

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

    self.optimizer = self.params.optimizer
    self.init_learning_rate = self.params.init_learning_rate
    self.num_epochs_per_decay = self.params.num_epochs_per_decay
    self.learning_rate_decay_factor = self.params.learning_rate_decay_factor
    self.minimum_learning_rate = self.params.minimum_learning_rate
    self.momentum = self.params.momentum
    self.rmsprop_decay = self.params.rmsprop_decay
    self.rmsprop_momentum = self.params.rmsprop_momentum
    self.rmsprop_epsilon = self.params.rmsprop_epsilon
    self.adam_beta1 = self.params.adam_beta1
    self.adam_beta2 = self.params.adam_beta2
    self.adam_epsilon = self.params.adam_epsilon
    self.use_fp16 = self.params.use_fp16
    self.fp16_vars = self.params.fp16_vars
    self.all_reduce_spec = self.params.all_reduce_spec
    self.save_checkpoints_steps = self.params.save_checkpoints_steps
    self.max_chkpts_to_keep = self.params.max_chkpts_to_keep
    self.output_dir = self.params.output_dir
    self.data_format = self.params.data_format

    if self.use_fp16 and self.fp16_vars:
      self.data_type = tf.float16
    else:
      self.data_type = tf.float32

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
    distribution_strategy = utils.get_distribution_strategy(
        self.num_gpus, self.all_reduce_spec)
    
    # Create session config. allow_soft_placement = True, is required for
    # multi-GPU and is not harmful for other modes.
    session_config = tf.ConfigProto(
        allow_soft_placement=True)

    # Create a 'RunConfig' object.
    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        save_checkpoints_steps=self.save_checkpoints_steps,
        keep_checkpoint_max=self.max_chkpts_to_keep,
        train_distribute=distribution_strategy)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn, 
        model_dir=self.output_dir,
        config=run_config,
        params=self.params)

    if not self.use_synthetic_data:
      input_function = datasets.input_fn
    else:
      input_function = datasets.get_synth_input_fn(self.data_type)

    def input_fn_train(num_epochs):
      return input_function(
              is_training=True,
              data_dir=self.data_dir,
              batch_size=self.batch_size_per_device,
              num_epochs=num_epochs,
              dtype=self.data_type,
              )

    def input_fn_eval():
      return input_function(
              is_training=False,
              data_dir=self.data_dir,
              batch_size=self.batch_size_per_device,
              num_epochs=1,
              dtype=self.data_type)

    time_hist = TimeHistory()

    if self.do_train:
      classifier.train(input_fn=lambda: input_fn_train(self.num_epochs), 
                       hooks=[time_hist]steps=1)
      total_time = sum(time_hist.times)
      print(f"Totoal time with {self.num_gpus} GPU(s): {total_time} seconds.")

      avg_time_per_batch = np.mean(time_hist.times)
      print(f"{self.batch/avg_time_per_batch} images/second.")
    else:
      classifier.evaluate(input_fn=lambda: input_fn_train(self.num_epochs))
