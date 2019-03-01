# --*-- coding:utf-8 --*--
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
import os

import tensorflow as tf

from models.cnn import alexnet_model as alexnet
from models.cnn import resnet_model as resnet
from models.cnn import googlenet_model as googlenet
from models.cnn import vgg_model as vgg
import imagenet_datasets as datasets
import utils

MODEL_CREATOR = {
        'alexnet': alexnet.AlexnetModel,
        'resnet50': resnet.create_resnet50_model,
        'googlenet': googlenet.GooglenetModel,
        'vgg16': vgg.Vgg16Model,
}

class ExamplesPerSecondHook(tf.train.SessionRunHook):
  """Hook to display examples per second."""

  def __init__(self,
               batch_size,
               every_n_steps=100):
    """Intializer.

    Args:
      batch_size: Total batch size across all workers.
      every_n_steps: Display the message every n steps.
    """
    self.timer = tf.train.SecondOrStepTimer(every_steps=every_n_steps)
    self.train_time = 0
    self.total_steps = 0
    self.batch_size = batch_size
    self.examples_per_second_list = []
  
  def begin(self):
    self.global_step = tf.train.get_global_step()
    if self.global_step == None:
      raise RuntimeError("Global step must be created before using this hook.")
  
  def before_run(self, run_context):
    return tf.train.SessionRunArgs(self.global_step)
  
  def after_run(self, run_context, run_values):
    global_step = run_values.results

    if self.timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self.timer.update_last_triggered_step(
              global_step)
      if elapsed_time is not None:
        self.total_steps += elapsed_steps
        self.train_time += elapsed_time
        average_examples_per_sec = self.batch_size * (
            self.total_steps / self.train_time)
        current_examples_per_sec = self.batch_size * (
            elapsed_steps / elapsed_time)
        self.examples_per_second_list.append(current_examples_per_sec)
        out_str = "average_examples_per_sec: {}".format(
                  average_examples_per_sec)
        tf.logging.info(out_str)
        out_str = "current_examples_per_sec: {}".format(
                  current_examples_per_sec)
        tf.logging.info(out_str)

class TimeHistory(tf.train.SessionRunHook):
  """Record the run time for each iteration of training/evaluation."""
  def begin(self):
    self.times = []
  
  def before_run(self, run_context):
    self.time_start = time.time()
  
  def after_run(self, run_context, run_values):
    self.times.append(time.time() - self.time_start)

def get_optimizer(params, learning_rate):
  """Returns the optimizer that should be used based on params."""
  if params.optimizer == 'momentum':
    opt = tf.train.MomentumOptimizer(
        learning_rate, params.momentum, use_nesterov=True)
  elif params.optimizer == 'sgd':
    opt = tf.train.GradientDescentOptimizer(learning_rate)
  elif params.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate, params.adam_beta1,
                                 params.adam_beta2, params.adam_epsilon)
  else:
    raise ValueError('Optimizer "{}" was not recognized'.
                     format(params.optimizer))
  return opt

def get_learning_rate(params, global_step):
  """Get a learning rate that decays step-wise."""
  if params.init_learning_rate is not None:
    boundaries = [1000 * x for x in [30, 60, 80]]
    values = [0.01, 0.001, 0.0001, 0.00001]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    return learning_rate
  else:
    raise ValueError("`--init_learning_rate` must be specified.")

def model_fn(features, labels, mode, params):
  """Define how to train, evaluate and predict."""
  tf.summary.image('images', features, max_outputs=6)
  with tf.variable_scope("model"):
    # Create model and get output logits.
    model = MODEL_CREATOR[params.model](params)

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

    # Calculate loss: softmax cross entropy.
    loss = tf.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)

    tf.identity(loss, name='loss')
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.EVAL:
      train_op = None

    elif mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_or_create_global_step()
      learning_rate = get_learning_rate(params, global_step)

      tf.identity(learning_rate, name='learning_rate')
      tf.summary.scalar('learning_rate', learning_rate)

      optimizer = get_optimizer(params, learning_rate)

      train_op = optimizer.minimize(loss, global_step=global_step)

      #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      #train_op = tf.group(train_op, update_ops)

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
    tf.logging.set_verbosity(tf.logging.INFO)
    self.params = params

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
    self.momentum = self.params.momentum
    self.adam_beta1 = self.params.adam_beta1
    self.adam_beta2 = self.params.adam_beta2
    self.adam_epsilon = self.params.adam_epsilon
    self.use_fp16 = self.params.use_fp16
    #self.fp16_vars = self.params.fp16_vars
    self.all_reduce_spec = self.params.all_reduce_spec
    self.save_checkpoints_steps = self.params.save_checkpoints_steps
    self.max_chkpts_to_keep = self.params.max_chkpts_to_keep
    self.model_dir = self.params.model_dir
    self.data_format = self.params.data_format

    #if self.use_fp16 and self.fp16_vars:
    #  self.data_type = tf.float16
    #else:
    #  self.data_type = tf.float32
    self.data_type = tf.float16 if self.params.use_fp16 else tf.float32

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
    print('Model:       %s' % self.params.model)
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
        model_dir=self.model_dir,
        config=run_config,
        params=self.params)

    if self.use_synthetic_data:
      input_function = datasets.get_synth_input_fn(self.data_type)
    else:
      input_function = datasets.input_fn

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

    for i in xrange(int(self.num_epochs)):
      train_hook = ExamplesPerSecondHook(self.params.batch_size)

      if self.do_train:
        classifier.train(input_fn=lambda: input_fn_train(self.num_epochs), 
                       hooks=[train_hook])
        total_time = train_hook.train_time
        print("Totoal time with {} GPU(s): {} seconds.".format(
            self.num_gpus, total_time))
        experments_per_sec_list = train_hook.examples_per_second_list
        #with open(os.path.join(self.params.output_dir, 
        #    self.params.model + '.txt'), 'w+') as f:
        #  for experiments_per_sec in experments_per_sec_list:
        #    line = "experiments_per_sec: " + str(experiments_per_sec) + '\n'
        #    f.writelines(line)
        cur_examples_per_sec = train_hook.examples_per_second_list[
                 len(train_hook.examples_per_second_list) - 1]
        print("Current examples per second: {}.".format(cur_examples_per_sec))

        #max_time_index = np.argmax(time_hist.times)
        #min_time_index = np.argmin(time_hist.times)
        #max_time = time_hist.times[max_time_index]
        #min_time = time_hist.times[min_time_index]
        #avg_time = np.mean(time_hist.times) # per batch
        #print("{} images/second (avg).".format(self.batch_size/avg_time))
        #print("{} images/second (max).".format(self.batch_size/max_time))
        #print("{} images/second (min).".format(self.batch_size/min_time))
      if self.do_eval:
        results = classifier.evaluate(input_fn=lambda: input_fn_eval())
        print("accuracy: {}, accuracy_top_5: {}".format(
            results['accuracy'], results['accuracy_top_5']))
