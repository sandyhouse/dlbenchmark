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
"""TensorFlow benchmark library.

See the README for more information.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import namedtuple
import contextlib
import math
import multiprocessing
import os
import re
import threading
import time
import traceback

from absl import flags as absl_flags
import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import cnn_util
import constants
import datasets
import flags
import variable_mgr
import variable_mgr_util
from cnn_util import log_fn
from models import model_config
from platforms import util as platforms_util
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_util_impl
from tensorflow.python.framework import importer
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest


_DEFAULT_NUM_BATCHES = 100


# GraphInfo encapsulates the tensors/ops that we care about after building a
# graph. We use them to benchmark the graph.
GraphInfo = namedtuple(  # pylint: disable=invalid-name
    'GraphInfo',
    [
        # Ops that produce the input batches (before preprocessing).
        'input_producer_op',
        # Ops that adds the preprocessed images to the staging areas
        'enqueue_ops',
        # Fetches of sess.run()
        'fetches',
        # Op that performs synchronization in distributed mode
        'execution_barrier',
        # The global step variable
        'global_step',
        # Group of ops that perform per-device initialization work
        'local_var_init_op_group',
        # Op to produce summaries
        'summary_op'
    ])


# InputProcessingInfo contains various sources of inputs which will be later fed
# into the model. If synthetic data is used, all three fields are None.
InputProcessingInfo = namedtuple(
    'InputProcessingInfo',
    [
        # The first two fields are non-None iff datasets prefetching is not
        # used.

        # Ops that produce the input batches.
        'input_producer_op',
        # A list of StagingArea for each device.
        'input_producer_stages',

        # Input produced using multi device iterator. Non-None iff datasets
        # prefetching is used
        'multi_device_iterator_input'
    ])


# TODO(reedwm): add upper_bound and lower_bound to appropriate integer and
# float flags, and change certain string flags to enum flags.
flags.DEFINE_string('model', 'trivial',
                    'Name of the model to run, the list of supported models '
                    'are defined in models/model.py')
flags.DEFINE_boolean('print_training_accuracy', False,
                     'whether to calculate and print training accuracy during '
                     'training')
flags.DEFINE_integer('batch_size', 0, 'batch size per compute device')
flags.DEFINE_integer('batch_group_size', 1,
                     'number of groups of batches processed in the image '
                     'producer.')
flags.DEFINE_integer('num_batches', None, 'number of batches to run, excluding '
                     'warmup. Defaults to %d' % _DEFAULT_NUM_BATCHES)
flags.DEFINE_float('num_epochs', None,
                   'number of epochs to run, excluding warmup. '
                   'This and --num_batches cannot both be specified.')
flags.DEFINE_integer('num_warmup_batches', None,
                     'number of batches to run before timing')
# TODO(tucker): change num_gpus to num_devices
flags.DEFINE_integer('num_gpus', 1, 'the number of GPUs to run on')
flags.DEFINE_integer('display_every', 10,
                     'Number of local steps after which progress is printed '
                     'out')
flags.DEFINE_string('data_dir', None,
                    'Path to dataset in TFRecord format (aka Example '
                    'protobufs). If not specified, synthetic data will be '
                    'used.')
flags.DEFINE_string('data_name', None,
                    'Name of dataset: imagenet or cifar10. If not specified, '
                    'it is automatically guessed based on data_dir.')
flags.DEFINE_string('resize_method', 'bilinear',
                    'Method for resizing input images: crop, nearest, '
                    'bilinear, bicubic, area, or round_robin. The `crop` mode '
                    'requires source images to be at least as large as the '
                    'network input size. The `round_robin` mode applies '
                    'different resize methods based on position in a batch in '
                    'a round-robin fashion. Other modes support any sizes and '
                    'apply random bbox distortions before resizing (even with '
                    'distortions=False).')
flags.DEFINE_boolean('distortions', True,
                     'Enable/disable distortions during image preprocessing. '
                     'These include bbox and color distortions.')
flags.DEFINE_boolean('use_datasets', True,
                     'Enable use of datasets for input pipeline')
flags.DEFINE_string('input_preprocessor', 'default',
                    'Name of input preprocessor. The list of supported input '
                    'preprocessors are defined in preprocessing.py.')
flags.DEFINE_string('gpu_thread_mode', 'gpu_private',
                    'Methods to assign GPU host work to threads. '
                    'global: all GPUs and CPUs share the same global threads; '
                    'gpu_private: a private threadpool for each GPU; '
                    'gpu_shared: all GPUs share the same threadpool.')
flags.DEFINE_integer('per_gpu_thread_count', 0,
                     'The number of threads to use for GPU. Only valid when '
                     'gpu_thread_mode is not global.')
flags.DEFINE_enum('variable_consistency', 'strong', ('strong', 'relaxed'),
                  'The data consistency for trainable variables. With strong '
                  'consistency, the variable always have the updates from '
                  'previous step. With relaxed consistency, all the updates '
                  'will eventually show up in the variables. Likely one step '
                  'behind.')
flags.DEFINE_enum('local_parameter_device', 'gpu', ('cpu', 'gpu', 'CPU', 'GPU'),
                  'Device to use as parameter server: cpu or gpu. For '
                  'distributed training, it can affect where caching of '
                  'variables happens.')
flags.DEFINE_enum('device', 'gpu', ('cpu', 'gpu', 'CPU', 'GPU'),
                  'Device to use for computation: cpu or gpu')
flags.DEFINE_enum('data_format', 'NCHW', ('NHWC', 'NCHW'),
                  'Data layout to use: NHWC (TF native) or NCHW (cuDNN '
                  'native, requires GPU).')
flags.DEFINE_string('trace_file', '',
                    'Enable TensorFlow tracing and write trace to this file.')
_NUM_STEPS_TO_PROFILE = 10
_NUM_OPS_TO_PRINT = 20
flags.DEFINE_string('tfprof_file', None,
                    'If specified, write a tfprof ProfileProto to this file. '
                    'The performance and other aspects of the model can then '
                    'be analyzed with tfprof. See '
                    'https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/command_line.md '  # pylint: disable=line-too-long
                    'for more info on how to do this. The first %d steps '
                    'are profiled. Additionally, the top %d most time '
                    'consuming ops will be printed.\n'
                    'Note: profiling with tfprof is very slow, but most of the '
                    'overhead is spent between steps. So, profiling results '
                    'are more accurate than the slowdown would suggest.' %
                    (_NUM_STEPS_TO_PROFILE, _NUM_OPS_TO_PRINT))
flags.DEFINE_enum('optimizer', 'adam', ('momentum', 'sgd', 'rmsprop', 'adam'),
                  'Optimizer to use')
flags.DEFINE_float('init_learning_rate', None,
                   'Initial learning rate for training.')
flags.DEFINE_float('num_epochs_per_decay', 0,
                   'Steps after which learning rate decays. If 0, the learning '
                   'rate does not decay.')
flags.DEFINE_float('learning_rate_decay_factor', 0,
                   'Learning rate decay factor. Decay by this factor every '
                   '`num_epochs_per_decay` epochs. If 0, learning rate does '
                   'not decay.')
flags.DEFINE_float('minimum_learning_rate', 0,
                   'The minimum learning rate. The learning rate will '
                   'never decay past this value. Requires `learning_rate`, '
                   '`num_epochs_per_decay` and `learning_rate_decay_factor` to '
                   'be set.')
flags.DEFINE_float('resnet_base_lr', None, "Base learning rate at bs=256. Only "
                   "relevant when training ResNet and utilizing the model's "
                   "learning rate heuristic (get_learning_rate).")
flags.DEFINE_float('momentum', 0.9, 'Momentum for training.')
flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum in RMSProp.')
flags.DEFINE_float('rmsprop_epsilon', 1.0, 'Epsilon term for RMSProp.')
flags.DEFINE_float('adam_beta1', 0.9, 'Beta1 term for the Adam optimizer')
flags.DEFINE_float('adam_beta2', 0.999, 'Beta2 term for the Adam optimizer')
flags.DEFINE_float('adam_epsilon', 1e-8, 'Epsilon term for the Adam optimizer')
flags.DEFINE_float('gradient_clip', None,
                   'Gradient clipping magnitude. Disabled by default.')
flags.DEFINE_float('weight_decay', 0.00004,
                   'Weight decay factor for training.')

flags.DEFINE_boolean('datasets_use_prefetch', True,
                     'Enable use of prefetched datasets for input pipeline. '
                     'This option is meaningless if use_datasets=False.')
flags.DEFINE_integer('datasets_prefetch_buffer_size', 1,
                     'Prefetching op buffer size per compute device.')
flags.DEFINE_integer('datasets_num_private_threads', None,
                     'Number of threads for a private threadpool created for '
                     'all datasets computation. By default, we pick an '
                     'appropriate number. If set to 0, we use the default '
                     'tf-Compute threads for dataset operations.')
flags.DEFINE_boolean('datasets_use_caching', False,
                     'Cache the compressed input data in memory. This improves '
                     'the data input performance, at the cost of additional '
                     'memory.')
flags.DEFINE_integer('datasets_parallel_interleave_cycle_length', None,
                     'Number of parallel file readers interleaving input data.')
flags.DEFINE_boolean('datasets_sloppy_parallel_interleave', False,
                     'Allow parallel interleave to depart from deterministic '
                     'ordering, by temporarily skipping over files whose '
                     'elements are not readily available. This can increase '
                     'througput in particular in the presence of stragglers.')
flags.DEFINE_integer('datasets_parallel_interleave_prefetch', None,
                     'The number of input elements to fetch before they are '
                     'needed for interleaving.')

# Performance tuning parameters.
flags.DEFINE_enum('loss_type_to_report', 'total_loss',
                  ('base_loss', 'total_loss'),
                  'Which type of loss to output and to write summaries for. '
                  'The total loss includes L2 loss while the base loss does '
                  'not. Note that the total loss is always used while '
                  'computing gradients during training if weight_decay > 0, '
                  'but explicitly computing the total loss, instead of just '
                  'computing its gradients, can have a performance impact.')
flags.DEFINE_boolean('single_l2_loss_op', False,
                     'If True, instead of using an L2 loss op per variable, '
                     'concatenate the variables into a single tensor and do a '
                     'single L2 loss on the concatenated tensor.')
flags.DEFINE_boolean('use_resource_vars', False,
                     'Use resource variables instead of normal variables. '
                     'Resource variables are slower, but this option is useful '
                     'for debugging their performance.')
# fp16 parameters. If use_fp16=False, no other fp16 parameters apply.
flags.DEFINE_boolean('use_fp16', False,
                     'Use 16-bit floats for certain tensors instead of 32-bit '
                     'floats. This is currently experimental.')
# TODO(reedwm): The default loss scale of 128 causes most models to diverge
# on the second step with synthetic data. Changing the tf.set_random_seed
# call to tf.set_random_seed(1235) or most other seed values causes the
# issue not to occur.
flags.DEFINE_float('fp16_loss_scale', None,
                   'If fp16 is enabled, the loss is multiplied by this amount '
                   'right before gradients are computed, then each gradient '
                   'is divided by this amount. Mathematically, this has no '
                   'effect, but it helps avoid fp16 underflow. Set to 1 to '
                   'effectively disable. Ignored during eval.')
flags.DEFINE_boolean('fp16_vars', False,
                     'If fp16 is enabled, also use fp16 for variables. If '
                     'False, the variables are stored in fp32 and casted to '
                     'fp16 when retrieved.  Recommended to leave as False.')
flags.DEFINE_boolean('fp16_enable_auto_loss_scale', False,
                     'If True and use_fp16 is True, automatically adjust the '
                     'loss scale during training.')
flags.DEFINE_integer('fp16_inc_loss_scale_every_n', 1000,
                     'If fp16 is enabled and fp16_enable_auto_loss_scale is '
                     'True, increase the loss scale every n steps.')

# The method for managing variables:
#   parameter_server: variables are stored on a parameter server that holds
#       the master copy of the variable. In local execution, a local device
#       acts as the parameter server for each variable; in distributed
#       execution, the parameter servers are separate processes in the
#       cluster.
#       For each step, each tower gets a copy of the variables from the
#       parameter server, and sends its gradients to the param server.
#   replicated: each GPU has its own copy of the variables. To apply
#       gradients, an all_reduce algorithm or or regular cross-device
#       aggregation is used to replicate the combined gradients to all
#       towers (depending on all_reduce_spec parameter setting).
#   independent: each GPU has its own copy of the variables, and gradients
#       are not shared between towers. This can be used to check performance
#       when no data is moved between GPUs.
#   distributed_replicated: Distributed training only. Each GPU has a copy
#       of the variables, and updates its copy after the parameter servers
#       are all updated with the gradients from all servers. Only works with
#       cross_replica_sync=true. Unlike 'replicated', currently never uses
#       nccl all-reduce for replicating within a server.
#   distributed_all_reduce: Distributed training where all replicas run
#       in a single session, using all-reduce to mutally reduce the
#       gradients.  Uses no parameter servers.  When there is only one
#       worker, this is the same as replicated.
#   collective_all_reduce: Distributed training where all replicas run
#       independepently except for variable initialization and for
#       gradient reduction which is done via collective all-reduce.
#       NOTE: collective_all_reduce in conjunction with use_fp16 can
#       lead to NaNs in some models (resnet50).  TODO(tucker): fix it.
#   horovod: Distributed training using Horovod library. Runs workers using
#       an MPI framework (e.g. Open MPI). Each worker runs training on
#       single GPU, and averages gradients using NCCL or MPI all-reduce.
#       See https://github.com/uber/horovod for more details.
flags.DEFINE_enum('variable_update', 'parameter_server',
                  ('parameter_server', 'replicated', 'distributed_replicated',
                   'independent', 'distributed_all_reduce',
                   'collective_all_reduce', 'horovod'),
                  'The method for managing variables: parameter_server, '
                  'replicated, distributed_replicated, independent, '
                  'distributed_all_reduce, collective_all_reduce, horovod')
flags.DEFINE_string('all_reduce_spec', None,
                    'A specification of the all_reduce algorithm to be used '
                    'for reducing gradients.  For more details, see '
                    'parse_all_reduce_spec in variable_mgr.py.  An '
                    'all_reduce_spec has BNF form:\n'
                    'int ::= positive whole number\n'
                    'g_int ::= int[KkMGT]?\n'
                    'alg_spec ::= alg | alg#int\n'
                    'range_spec ::= alg_spec | alg_spec/alg_spec\n'
                    'spec ::= range_spec | range_spec:g_int:range_spec\n'
                    'NOTE: not all syntactically correct constructs are '
                    'supported.\n\n'
                    'Examples:\n '
                    '"xring" == use one global ring reduction for all '
                    'tensors\n'
                    '"pscpu" == use CPU at worker 0 to reduce all tensors\n'
                    '"nccl" == use NCCL to locally reduce all tensors.  '
                    'Limited to 1 worker.\n'
                    '"nccl/xring" == locally (to one worker) reduce values '
                    'using NCCL then ring reduce across workers.\n'
                    '"pscpu:32k:xring" == use pscpu algorithm for tensors of '
                    'size up to 32kB, then xring for larger tensors.')

# Distributed training parameters.
flags.DEFINE_enum('job_name', '', ('ps', 'worker', 'controller', ''),
                  'One of "ps", "worker", "controller", "".  Empty for local '
                  'training')
flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of target hosts')
flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of target hosts')
flags.DEFINE_string('controller_host', None, 'optional controller host')
flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')
flags.DEFINE_boolean('cross_replica_sync', True, '')
flags.DEFINE_string('horovod_device', '', 'Device to do Horovod all-reduce on: '
                    'empty (default), cpu or gpu. Default with utilize GPU if '
                    'Horovod was compiled with the HOROVOD_GPU_ALLREDUCE '
                    'option, and CPU otherwise.')

# Summary and Save & load checkpoints.
flags.DEFINE_integer('summary_verbosity', 0, 'Verbosity level for summary ops. '
                     'level 0: disable any summary.\n'
                     'level 1: small and fast ops, e.g.: learning_rate, '
                     'total_loss.\n'
                     'level 2: medium-cost ops, e.g. histogram of all '
                     'gradients.\n'
                     'level 3: expensive ops: images and histogram of each '
                     'gradient.\n')
flags.DEFINE_integer('save_summaries_steps', 0,
                     'How often to save summaries for trained models. Pass 0 '
                     'to disable summaries.')
flags.DEFINE_integer('save_model_secs', 0,
                     'How often to save trained models. Pass 0 to disable '
                     'saving checkpoints every N seconds. A checkpoint is '
                     'saved after training completes regardless of this '
                     'option.')
flags.DEFINE_integer('save_model_steps', None,
                     'How often to save trained models. If specified, '
                     'save_model_secs must not be specified.')
flags.DEFINE_integer('max_ckpts_to_keep', 5,
                     'Max number of checkpoints to keep.')
flags.DEFINE_string('train_dir', None,
                    'Path to session checkpoints. Pass None to disable saving '
                    'checkpoint at the end.')

class GlobalStepWatcher(threading.Thread):
  """A helper class for global_step.

  Polls for changes in the global_step of the model, and finishes when the
  number of steps for the global run are done.
  """

  def __init__(self, sess, global_step_op, start_at_global_step,
               end_at_global_step):
    threading.Thread.__init__(self)
    self.sess = sess
    self.global_step_op = global_step_op
    self.start_at_global_step = start_at_global_step
    self.end_at_global_step = end_at_global_step

    self.start_time = 0
    self.start_step = 0
    self.finish_time = 0
    self.finish_step = 0

  def run(self):
    while self.finish_time == 0:
      time.sleep(.25)
      global_step_val, = self.sess.run([self.global_step_op])
      if self.start_time == 0 and global_step_val >= self.start_at_global_step:
        # Use tf.logging.info instead of log_fn, since print (which is log_fn)
        # is not thread safe and may interleave the outputs from two parallel
        # calls to print, which can break tests.
        tf.logging.info('Starting real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.start_time = time.time()
        self.start_step = global_step_val
      if self.finish_time == 0 and global_step_val >= self.end_at_global_step:
        tf.logging.info('Finishing real work at step %s at time %s' %
                        (global_step_val, time.ctime()))
        self.finish_time = time.time()
        self.finish_step = global_step_val

  def done(self):
    return self.finish_time > 0

  def num_steps(self):
    return self.finish_step - self.start_step

  def elapsed_time(self):
    return self.finish_time - self.start_time


def create_config_proto(params):
  """Returns session config proto.

  Args:
    params: Params tuple, created by make_params_from_flags.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  if params.device == 'gpu':
    config.intra_op_parallelism_threads = 1
  config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
  if params.device == 'cpu':
    config.device_count['CPU'] = params.num_gpus
  if params.variable_update == 'collective_all_reduce':
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.scoped_allocator_optimization = (
        rewriter_config_pb2.RewriterConfig.ON)
    rewrite_options.scoped_allocator_opts.enable_op.append('CollectiveReduce')
  if params.variable_update == 'horovod':
    import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
    config.gpu_options.visible_device_list = str(hvd.local_rank())
  # For collective_all_reduce, ignore all devices except current worker.
  if params.variable_update == 'collective_all_reduce':
    del config.device_filters[:]
    config.device_filters.append(
        '/job:%s/replica:0/task:%d' % (params.job_name, params.task_index))

  return config


# How many digits to show for the loss and accuracies during training.
LOSS_AND_ACCURACY_DIGITS_TO_SHOW = 3

def benchmark_one_step(sess,
                       fetches,
                       step,
                       batch_size,
                       step_train_times,
                       trace_filename,
                       profiler,
                       image_producer,
                       params,
                       summary_op=None,
                       show_images_per_sec=True,
                       collective_graph_key=0):
  """Advance one step of benchmarking."""
  should_profile = profiler and 0 <= step < _NUM_STEPS_TO_PROFILE
  need_options_and_metadata = (
      should_profile or collective_graph_key > 0 or
      (trace_filename and step == -2)
  )
  if need_options_and_metadata:
    run_options = tf.RunOptions()
    if (trace_filename and step == -2) or should_profile:
      run_options.trace_level = tf.RunOptions.FULL_TRACE
    if collective_graph_key > 0:
      run_options.experimental.collective_graph_key = collective_graph_key
    run_metadata = tf.RunMetadata()
  else:
    run_options = None
    run_metadata = None
  summary_str = None
  start_time = time.time()
  if summary_op is None:
    results = sess.run(fetches, options=run_options, run_metadata=run_metadata)
  else:
    (results, summary_str) = sess.run(
        [fetches, summary_op], options=run_options, run_metadata=run_metadata)

  lossval = results['average_loss']
  if image_producer is not None:
    image_producer.notify_image_consumption()
  train_time = time.time() - start_time
  step_train_times.append(train_time)
  if (show_images_per_sec and step >= 0 and
      (step == 0 or (step + 1) % params.display_every == 0)):
    speed_mean, speed_uncertainty, speed_jitter = get_perf_timing(
        batch_size, step_train_times)
    log_str = '%i\t%s\t%.*f' % (
        step + 1,
        get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter),
        LOSS_AND_ACCURACY_DIGITS_TO_SHOW, lossval)
    if 'top_1_accuracy' in results:
      log_str += '\t%.*f\t%.*f' % (
          LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_1_accuracy'],
          LOSS_AND_ACCURACY_DIGITS_TO_SHOW, results['top_5_accuracy'])
    log_fn(log_str)
  if need_options_and_metadata:
    if should_profile:
      profiler.add_step(step, run_metadata)
    if trace_filename and step == -2:
      log_fn('Dumping trace to %s' % trace_filename)
      trace_dir = os.path.dirname(trace_filename)
      if not gfile.Exists(trace_dir):
        gfile.MakeDirs(trace_dir)
      with gfile.Open(trace_filename, 'w') as trace_file:
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
  return (summary_str, lossval)

def get_perf_timing_str(speed_mean, speed_uncertainty, speed_jitter):
  return ('images/sec: %.1f +/- %.1f (jitter = %.1f)' %
            (speed_mean, speed_uncertainty, speed_jitter))

def get_perf_timing(batch_size, step_train_times):
  """Calculate benchmark processing speed."""
  times = np.array(step_train_times)
  speeds = batch_size / times
  time_mean = np.mean(times)
  speed_mean = batch_size / time_mean
  speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
  speed_jitter = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
  return speed_mean, speed_uncertainty, speed_jitter

# Params are passed to BenchmarkCNN's constructor. Params is a map from name
# to value, with one field per key in flags.param_specs.
#
# Call make_params_from_flags() below to construct a Params
# tuple with default values from flags.param_specs, rather than constructing
# Params directly.
Params = namedtuple('Params', flags.param_specs.keys())  # pylint: disable=invalid-name

def make_params_from_flags():
  """Create a Params tuple for BenchmarkCNN from absl_flags.FLAGS.

  Returns:
    Params namedtuple for constructing BenchmarkCNN.
  """
  # Collect (name: value) pairs for absl_flags.FLAGS with matching names in
  # flags.param_specs.
  flag_values = {name: getattr(absl_flags.FLAGS, name)
                 for name in flags.param_specs.keys()}
  return Params(**flag_values)

def get_num_batches_and_epochs(params, batch_size, num_examples_per_epoch):
  """Returns the number of batches and epochs to run for.

  Args:
    params: Params tuple, typically created by make_params_from_flags.
    batch_size: The number of images per step.
    num_examples_per_epoch: The number of images in a single epoch.

  Returns:
    num_batches: The number of batches to run for.
    num_epochs: The number of epochs to run for. This might be slightly
      smaller than params.num_epochs if specified, because the number of batches
      must be an integer.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  if params.num_batches and params.num_epochs:
    raise ValueError('At most one of --num_batches and --num_epochs may be '
                     'specified.')
  if params.num_epochs:
    num_batches = int(params.num_epochs * num_examples_per_epoch +
                      batch_size - 1) // batch_size
  else:
    num_batches = params.num_batches or _DEFAULT_NUM_BATCHES
  num_epochs = num_batches * batch_size / num_examples_per_epoch
  return (num_batches, num_epochs)


def get_learning_rate(params, global_step, num_examples_per_epoch, model,
                      batch_size):
  """Returns a learning rate tensor based on global_step.

  Args:
    params: Params tuple created by make_params_from_flags.
    global_step: Scalar tensor representing the global step.
    num_examples_per_epoch: The number of examples per epoch.
    model: The model.Model object to obtain the default learning rate from if no
      learning rate is specified.
    batch_size: Number of examples per step

  Returns:
    A scalar float tensor, representing the learning rate. When evaluated, the
    learning rate depends on the current value of global_step.

  Raises:
    ValueError: Invalid or unsupported params.
  """
  with tf.name_scope('learning_rate'):
    num_batches_per_epoch = num_examples_per_epoch / batch_size

    if params.init_learning_rate is not None:
      learning_rate = params.init_learning_rate
      if (params.num_epochs_per_decay > 0 and
          params.learning_rate_decay_factor > 0):
        decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)

        # Decay the learning rate exponentially based on the number of steps.
        learning_rate = tf.train.exponential_decay(
            params.init_learning_rate,
            global_step,
            decay_steps,
            params.learning_rate_decay_factor,
            staircase=True)

        if params.minimum_learning_rate != 0.:
          learning_rate = tf.maximum(learning_rate,
                                     params.minimum_learning_rate)
    else:
      learning_rate = model.get_learning_rate(global_step, batch_size)

  return learning_rate


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


def generate_tfprof_profile(profiler, tfprof_file):
  """Generates a tfprof profile, writing it to a file and printing top ops.

  Args:
    profiler: A tf.profiler.Profiler. `profiler.add_step` must have already been
      called.
    tfprof_file: The filename to write the ProfileProto to.
  """
  profile_proto = profiler.serialize_to_string()
  log_fn('Dumping ProfileProto to %s' % tfprof_file)
  with gfile.Open(tfprof_file, 'wb') as f:
    f.write(profile_proto)

  # Print out the execution times of the top operations. Note this
  # information can also be obtained with the dumped ProfileProto, but
  # printing it means tfprof doesn't have to be used if all the user wants
  # is the top ops.
  options = tf.profiler.ProfileOptionBuilder.time_and_memory()
  options['max_depth'] = _NUM_OPS_TO_PRINT
  options['order_by'] = 'accelerator_micros'
  profiler.profile_operations(options)


class BenchmarkCNN(object):
  """Class for benchmarking a cnn network."""

  def __init__(self, params, dataset=None, model=None):
    """Initialize BenchmarkCNN.

    Args:
      params: Params tuple, created by make_params_from_flags.
      dataset: If not None, the dataset to use. Otherwise, params is used to
               obtain the dataset.
      model: If not None, the model to use. Otherwise, params is used to obtain
             the model.
    Raises:
      ValueError: Unsupported params settings.
    """
    self.params = params
    self.dataset = dataset or datasets.create_dataset(self.params.data_dir,
                                                      self.params.data_name)
    self.model = model or model_config.get_model_config(
        self.params.model, self.dataset, self.params)
    self.trace_filename = self.params.trace_file
    min_autotune_warmup = 5
    self.num_warmup_batches = self.params.num_warmup_batches if (
        self.params.num_warmup_batches is not None) else max(
            10, min_autotune_warmup)
    self.resize_method = self.params.resize_method
    self.sync_queue_counter = 0
    self.num_gpus = self.params.num_gpus

    if (self.params.device == 'cpu' and self.params.data_format == 'NCHW'):
      raise ValueError('device=cpu requires that data_format=NHWC')

    if ((self.params.num_epochs_per_decay or
         self.params.learning_rate_decay_factor) and
        not (self.params.init_learning_rate is not None and
             self.params.num_epochs_per_decay
             and self.params.learning_rate_decay_factor)):
      raise ValueError('If one of num_epochs_per_decay or '
                       'learning_rate_decay_factor is set, both must be set'
                       'and learning_rate must be set')
    if (self.params.minimum_learning_rate and
        not (self.params.init_learning_rate is not None and
             self.params.num_epochs_per_decay and
             self.params.learning_rate_decay_factor)):
      raise ValueError('minimum_learning_rate requires learning_rate,'
                       'num_epochs_per_decay, and '
                       'learning_rate_decay_factor to be set')

    if (self.params.use_fp16 and self.params.fp16_vars and
        'replicated' in self.params.variable_update and
        self.params.all_reduce_spec and 'nccl' in self.params.all_reduce_spec):
      raise ValueError('fp16 variables are not supported with NCCL')

    if self.params.variable_update == 'horovod' and self.params.num_gpus > 1:
      raise ValueError('Horovod benchmarks require num_gpus=1 on each worker')

    if self.params.variable_update == 'horovod' and self.params.job_name:
      raise ValueError('job_name should not be specified for Horovod.')

    if self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale:
      if self.params.all_reduce_spec and 'nccl' in self.params.all_reduce_spec:
        raise ValueError('Automatic loss scaling is not supported with NCCL.')
      if self.params.variable_update not in ('parameter_server', 'replicated',
                                             'independent'):
        raise ValueError('Automatic loss scaling is not supported with '
                         'variable_update=%s.' % self.params.variable_update)

    if params.save_model_secs and params.save_model_steps:
      raise ValueError('At most one of --save_model_secs and '
                       '--save_model_steps can be specified')

    self.mode = constants.BenchmarkMode.TRAIN

    # Use the batch size from the command line if specified, otherwise use the
    # model's default batch size.  Scale the benchmark's batch size by the
    # number of GPUs.
    if self.params.batch_size > 0:
      self.model.set_batch_size(self.params.batch_size)
    self.batch_size = self.model.get_batch_size() * self.num_gpus
    self.train_batch_size = self.batch_size
    self.batch_group_size = self.params.batch_group_size
    self.enable_auto_loss_scale = (
        self.params.use_fp16 and self.params.fp16_enable_auto_loss_scale)
    self.loss_scale = None
    self.loss_scale_normal_steps = None

    self.job_name = self.params.job_name  # "" for local training

    # PS server is used for distributed jobs not using all-reduce.
    use_ps_server = self.job_name and (self.params.variable_update !=
                                       'distributed_all_reduce' and
                                       self.params.variable_update !=
                                       'collective_all_reduce')
    # controller is used for distributed_all_reduce with > 1 worker.
    use_controller = (
        self.params.variable_update == 'distributed_all_reduce' and
        self.job_name)
    if use_controller and not params.controller_host:
      raise ValueError('When variable_update==distributed_all_reduce '
                       'controller_host must also be specified.')
    # collective_all_reduce doesn't need a controller or ps
    self.distributed_collective = (
        self.params.variable_update == 'collective_all_reduce' and
        self.job_name)

    self.local_parameter_device_flag = self.params.local_parameter_device
    if self.job_name:
      self.task_index = self.params.task_index
      self.cluster_manager = platforms_util.get_cluster_manager(
          params, create_config_proto(params))
      assert isinstance(self.cluster_manager, cnn_util.BaseClusterManager)

      worker_prefix = '/job:worker/replica:0/task:%s' % self.task_index
      if use_ps_server:
        self.param_server_device = tf.train.replica_device_setter(
            worker_device=worker_prefix + '/cpu:0',
            cluster=self.cluster_manager.get_cluster_spec())
        # This device on which the queues for managing synchronization between
        # servers should be stored.
        self.sync_queue_devices = [
            '/job:ps/replica:0/task:%s/cpu:0' % i
            for i in range(self.cluster_manager.num_ps())
        ]
      else:
        self.sync_queue_devices = ['/job:worker/replica:0/task:0/cpu:0']
      log_fn("task_index: {}".format(self.task_index))
      log_fn("sync_queue_devices: {}".format(self.sync_queue_devices))
      log_fn("param_server_device: {}".format(self.param_server_device))
    else:
      self.task_index = 0
      self.cluster_manager = None
      worker_prefix = ''
      self.param_server_device = '/%s:0' % self.params.local_parameter_device
      self.sync_queue_devices = [self.param_server_device]
      log_fn("task_index: {}".format(self.task_index))
      log_fn("sync_queue_devices: {}".format(self.sync_queue_devices))
      log_fn("param_server_device: {}".format(self.param_server_device))

    if self.cluster_manager:
      self.num_workers = self.cluster_manager.num_workers()
    elif self.params.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      self.num_workers = hvd.size()
    else:
      self.num_workers = 1
    self.num_ps = self.cluster_manager.num_ps() if self.cluster_manager else 0
    log_fn("num_workers: {}, num_ps: {}".format(self.num_workers, self.num_ps))

    if self.num_workers > 1 and self.params.all_reduce_spec == 'nccl':
      raise ValueError('--all_reduce_spec=nccl is invalid in a '
                       'multi-worker job')

    # Device to use for ops that need to always run on the local worker's CPU.
    self.cpu_device = '%s/cpu:0' % worker_prefix

    # Device to use for ops that need to always run on the local worker's
    # compute device, and never on a parameter server device.
    self.raw_devices = [
        '%s/%s:%i' % (worker_prefix, self.params.device, i)
        for i in xrange(self.num_gpus)
    ]

    subset = 'train'
    self.num_batches, self.num_epochs = get_num_batches_and_epochs(
        params, self.batch_size * self.num_workers,
        self.dataset.num_examples_per_epoch(subset))

    num_train_examples_per_epoch = self.dataset.num_examples_per_epoch('train')

    if self.params.variable_update == 'parameter_server':
      if self.job_name:
        self.variable_mgr = variable_mgr.VariableMgrDistributedFetchFromPS(
              self)
      else:
        self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
    elif self.params.variable_update == 'replicated':
      if self.job_name:
        raise ValueError('Invalid variable_update in distributed mode: %s' %
                         self.params.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
          self, self.params.all_reduce_spec, 0, 10, 1)
    elif self.params.variable_update == 'distributed_all_reduce':
      assert self.params.cross_replica_sync
      self.variable_mgr = variable_mgr.VariableMgrDistributedAllReduce(
          self, self.params.all_reduce_spec,
          ('worker' if self.num_workers > 1 else 'localhost'),
          self.num_workers, 0, 10, 1)
    elif self.params.variable_update == 'collective_all_reduce':
      assert self.params.cross_replica_sync
      self.variable_mgr = variable_mgr.VariableMgrCollectiveAllReduce(
          self, self.params.all_reduce_spec,
          self.num_workers, self.num_gpus, self.task_index,
          1)
    elif self.params.variable_update == 'distributed_replicated':
      assert self.params.cross_replica_sync
      if not self.job_name:
        raise ValueError('Invalid variable_update in local mode: %s' %
                         self.params.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrDistributedReplicated(self)
    elif self.params.variable_update in ('independent', 'horovod'):
      if self.job_name:
        raise ValueError('Invalid variable_update in distributed mode: %s' %
                         self.params.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrIndependent(self)
    else:
      raise ValueError(
          'Invalid variable_update: %s' % self.params.variable_update)

    # Device to use for running on the local worker's compute device, but
    # with variables assigned to parameter server devices.
    self.devices = self.variable_mgr.get_devices()
    if self.job_name:
      if use_ps_server:
        self.global_step_device = self.param_server_device
      elif self.params.variable_update == 'collective_all_reduce':
        self.global_step_device = self.cpu_device
      else:
        self.global_step_device = '/job:worker/replica:0/task:0/cpu:0'
    else:
      self.global_step_device = self.cpu_device

    self.input_preprocessor = None
    if not self.dataset.use_synthetic_gpu_inputs():
      self.input_preprocessor = self.get_input_preprocessor()
    self.datasets_use_prefetch = (
        self.params.datasets_use_prefetch and
        # TODO(rohanj): Figure out why --datasets_use_prefetch freezes on the
        # CPU.
        self.params.device.lower() != 'cpu' and
        self.input_preprocessor and
        self.input_preprocessor.supports_datasets())
    self.init_global_step = 0

  # TODO(laigd): this changes the global device list which is used everywhere,
  # consider refactoring it.
  def reset_devices_for_task(self, task_num, is_local=False):
    """Used to imitate another task when building a distributed graph."""
    worker_prefix = ('/job:localhost' if is_local else
                     '/job:worker/replica:0/task:%s' % task_num)
    self.cpu_device = '%s/cpu:0' % worker_prefix
    self.raw_devices = [
        '%s/%s:%i' % (worker_prefix, self.params.device, i)
        for i in xrange(self.num_gpus)
    ]
    self.devices = self.variable_mgr.get_devices()

  def raw_devices_across_tasks(self, is_local=False):
    """Returns list of raw device names across all tasks."""
    if is_local:
      assert self.num_workers == 1
      return self.raw_devices
    else:
      return [
          'job:worker/replica:0/task%s/%s:%i' % (t, self.params.device, i)
          for t in xrange(self.num_workers)
          for i in xrange(self.num_gpus)
      ]

  def print_info(self):
    """Print basic information."""
    benchmark_info = self._get_params_info()
    log_fn('Model:       %s' % self.model.get_model_name())
    log_fn('Dataset:     %s' % benchmark_info['dataset_name'])
    log_fn('Mode:        %s' % self.mode)
    log_fn('SingleSess:  %s' % benchmark_info['single_session'])
    log_fn('Batch size:  %s global' % (self.batch_size * self.num_workers))
    log_fn('             %s per device' % (self.batch_size //
                                           len(self.raw_devices)))
    if self.batch_group_size > 1:
      log_fn('             %d batches per prepocessing group' %
             self.batch_group_size)
    log_fn('Num batches: %d' % self.num_batches)
    log_fn('Num epochs:  %.2f' % self.num_epochs)
    log_fn('Devices:     %s' % benchmark_info['device_list'])
    log_fn('Data format: %s' % self.params.data_format)
    log_fn('Optimizer:   %s' % self.params.optimizer)
    log_fn('Variables:   %s' % self.params.variable_update)
    if (self.params.variable_update == 'replicated' or
        self.params.variable_update == 'distributed_all_reduce'
        or self.params.variable_update == 'collective_all_reduce'):
      log_fn('AllReduce:   %s' % self.params.all_reduce_spec)
    if self.job_name:
      log_fn('Sync:        %s' % self.params.cross_replica_sync)
    if self.params.variable_update == 'horovod' and self.params.horovod_device:
      log_fn('Horovod on:  %s' % self.params.horovod_device)
    log_fn('==========')

  def _get_params_info(self):
    """Get the common parameters info for the benchmark run.

    Returns:
      A dict of processed parameters.
    """
    dataset_name = self.dataset.name
    if self.dataset.use_synthetic_gpu_inputs():
      dataset_name += ' (synthetic)'
    single_session = self.params.variable_update == 'distributed_all_reduce'
    if single_session:
      device_list = self.raw_devices_across_tasks()
    elif self.params.variable_update == 'horovod':
      device_list = ['horovod/%s:%d' % (self.params.device, idx)
                     for idx in range(self.num_workers)]
    else:
      device_list = self.raw_devices
    return {
        'dataset_name': dataset_name,
        'single_session': single_session,
        'device_list': device_list,}

  def run(self):
    """Run the benchmark task assigned to this process.

    Returns:
      Dictionary of statistics for training or eval.
    Raises:
       ValueError: unrecognized job name.
    """
    if self.params.job_name == 'ps':
      log_fn('Running parameter server %s' % self.task_index)
      self.cluster_manager.join_server()
      return {}

    # For distributed_all_reduce with multiple workers, drive
    # from a separate controller process.
    if self.params.variable_update == 'distributed_all_reduce':
      if self.params.job_name == 'worker':
        log_fn('Starting worker %s' % self.task_index)
        self.cluster_manager.join_server()
        return
      elif self.params.job_name and self.params.job_name != 'controller':
        raise ValueError('unrecognized job name: %s' % self.params.job_name)

    return self._benchmark_train()

  def _benchmark_train(self):
    """Run cnn in benchmark mode.

    Returns:
      Dictionary containing training statistics (num_workers, num_steps,
      average_wall_time, images_per_sec).
    """
    graph = tf.Graph()
    with graph.as_default():
      build_result = self._build_graph()
    with graph.as_default():
      return self._benchmark_graph(build_result, None)

  GPU_CACHED_INPUT_VARIABLE_NAME = 'gpu_cached_inputs'

  def _build_graph(self):
    """Build the graph.

    Returns:
      A namedtuple containing the ops/tensors that required by
      _benchmark_graph().
    """
    if self.params.variable_update == 'distributed_all_reduce':
      self.single_session = True
      (input_producer_op, enqueue_ops, fetches) = (
          self._build_model_single_session())
    else:
      self.single_session = False
      (input_producer_op, enqueue_ops, fetches) = self._build_model()
    fetches_list = nest.flatten(list(fetches.values()))
    main_fetch_group = tf.group(*fetches_list, name='main_fetch_group')
    execution_barrier = None
    if (not self.single_session and self.job_name and
        not self.params.cross_replica_sync):
      execution_barrier = self.add_sync_queues_and_barrier(
          'execution_barrier_', [])

    global_step = tf.train.get_global_step()
    with tf.device(self.global_step_device), tf.name_scope('inc_global_step'):
      with tf.control_dependencies([main_fetch_group]):
        fetches['inc_global_step'] = global_step.assign_add(1)

    if ((not self.single_session) and (not self.distributed_collective) and
        self.job_name and self.params.cross_replica_sync):
      # Block all replicas until all replicas are ready for next step.
      fetches['sync_queues'] = self.add_sync_queues_and_barrier(
          'sync_queues_step_end_', [main_fetch_group])

    # Skips the init ops for freezable local variables in forward_only mode so
    # we can remove all the assign ops when converting variables to constants.
    with tf.name_scope('local_variable_initialization'):
      local_var_init_op = tf.local_variables_initializer()
    table_init_ops = tf.tables_initializer()

    variable_manager_init_ops = [local_var_init_op]
    if table_init_ops:
      variable_manager_init_ops.extend([table_init_ops])
    with tf.control_dependencies([local_var_init_op]):
      variable_manager_init_ops.extend(self.variable_mgr.get_post_init_ops())
    if ((not self.single_session) and (not self.distributed_collective) and
        self.job_name and self.params.cross_replica_sync):
      # Ensure all workers execute variable_manager_init_ops before they start
      # executing the model.
      variable_manager_init_ops.append(
          self.add_sync_queues_and_barrier('init_ops_end_',
                                           variable_manager_init_ops))
    local_var_init_op_group = tf.group(*variable_manager_init_ops,
                                       name='local_var_init_op_group')
    summary_op = tf.summary.merge_all()

    return GraphInfo(
        input_producer_op=input_producer_op,
        enqueue_ops=enqueue_ops,
        fetches=fetches,
        execution_barrier=execution_barrier,
        global_step=global_step,
        local_var_init_op_group=local_var_init_op_group,
        summary_op=summary_op)

  def _benchmark_graph(self, graph_info, eval_graph_info=None):
    """Benchmark the training graph.

    Args:
      graph_info: the namedtuple returned by _build_graph() which
        contains all necessary information to benchmark the graph, including
        named tensors/ops list, fetches, etc.
      eval_graph_info: Similar to graph_info but for the eval graph if
        --eval_during_training_* is used. Otherwise, None.
    Returns:
      Dictionary containing training statistics (num_workers, num_steps,
      average_wall_time, images_per_sec).
    """
    log_fn('Initializing graph')
    if self.params.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      # First worker will be 'chief' - it will write summaries and
      # save checkpoints.
      is_chief = hvd.rank() == 0
    else:
      is_chief = (not self.job_name or self.task_index == 0)

    summary_writer = None
    if (is_chief and self.params.summary_verbosity and self.params.train_dir and
        self.params.save_summaries_steps > 0):
      summary_writer = tf.summary.FileWriter(self.params.train_dir,
                                             tf.get_default_graph())

    # We want to start the benchmark timer right after a image_producer barrier
    # and avoids undesired waiting times on barriers.
    if ((self.num_warmup_batches + len(graph_info.enqueue_ops) - 1) %
        self.batch_group_size) != 0:
      self.num_warmup_batches = int(
          math.ceil(
              (self.num_warmup_batches + len(graph_info.enqueue_ops) - 1.0) /
              (self.batch_group_size)) * self.batch_group_size -
          len(graph_info.enqueue_ops) + 1)
      log_fn('Round up warm up steps to %d to match batch_group_size' %
             self.num_warmup_batches)
      assert ((self.num_warmup_batches + len(graph_info.enqueue_ops) - 1) %
              self.batch_group_size) == 0
    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    if is_chief:
      saver = tf.train.Saver(
          self.variable_mgr.savable_variables(),
          save_relative_paths=True,
          max_to_keep=self.params.max_ckpts_to_keep)
    else:
      saver = None
    ready_for_local_init_op = None
    if self.job_name and not (self.single_session or
                              self.distributed_collective):
      # In distributed mode, we don't want to run local_var_init_op_group until
      # the global variables are initialized, because local_var_init_op_group
      # may use global variables (such as in distributed replicated mode). We
      # don't set this in non-distributed mode, because in non-distributed mode,
      # local_var_init_op_group may itself initialize global variables (such as
      # in replicated mode).
      ready_for_local_init_op = tf.report_uninitialized_variables(
          tf.global_variables())
    if self.params.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      bcast_global_variables_op = hvd.broadcast_global_variables(0)
    else:
      bcast_global_variables_op = None

    if self.params.variable_update == 'collective_all_reduce':
      # It doesn't matter what this collective_graph_key value is,
      # so long as it's > 0 and the same at every worker.
      init_run_options = tf.RunOptions()
      init_run_options.experimental.collective_graph_key = 6
    else:
      init_run_options = tf.RunOptions()
    local_var_init_ops = [graph_info.local_var_init_op_group]
    sv = tf.train.Supervisor(
        # For the purpose of Supervisor, all Horovod workers are 'chiefs',
        # since we want session to be initialized symmetrically on all the
        # workers.
        is_chief=is_chief or (self.params.variable_update == 'horovod'
                              or self.distributed_collective),
        # Log dir should be unset on non-chief workers to prevent Horovod
        # workers from corrupting each other's checkpoints.
        logdir=self.params.train_dir if is_chief else None,
        ready_for_local_init_op=ready_for_local_init_op,
        local_init_op=local_var_init_ops,
        saver=saver,
        global_step=graph_info.global_step,
        summary_op=None,
        save_model_secs=self.params.save_model_secs,
        summary_writer=summary_writer,
        local_init_run_options=init_run_options)

    profiler = tf.profiler.Profiler() if self.params.tfprof_file else None

    start_standard_services = (
        self.params.train_dir or
        self.dataset.queue_runner_required())
    target = self.cluster_manager.get_target() if self.cluster_manager else ''
    with sv.managed_session(
        master=target,
        config=create_config_proto(self.params),
        start_standard_services=start_standard_services) as sess:
      # Anything that can potentially raise an OutOfRangeError with 'sess' MUST
      # be under this try block. The managed_session() context manager silently
      # ignores OutOfRangeError, so we must catch them and wrap them with
      # a different exception type so that they can be propagated up to the
      # caller.
      try:
        stats = self.benchmark_with_session(
            sess, sv, graph_info, eval_graph_info, bcast_global_variables_op,
            is_chief, summary_writer, profiler)
      except tf.errors.OutOfRangeError:
        raise RuntimeError(
            'Received OutOfRangeError. Wrapping in Runtime error to avoid '
            'Supervisor from suppressing the error. Original OutOfRangeError '
            'with traceback:\n' + traceback.format_exc())

    sv.stop()
    if profiler:
      generate_tfprof_profile(profiler, self.params.tfprof_file)
    return stats

  def benchmark_with_session(self, sess, supervisor, graph_info,
                             eval_graph_info, bcast_global_variables_op,
                             is_chief, summary_writer, profiler):
    """Benchmarks the graph with the given session.

    Args:
      sess: The session to benchmark the graph with
      supervisor: The Supervisor that created the session.
      graph_info: the namedtuple returned by _build_graph() which
        contains all necessary information to benchmark the graph, including
        named tensors/ops list, fetches, etc.
      eval_graph_info: Similar to graph_info but for the eval graph if
        --eval_during_training_every_n_steps is used. Otherwise, None.
      bcast_global_variables_op: If Horovod is used, the op to broadcast the
        global variables to all the processes. None if Horovod is not used.
      is_chief: True if this is the chief process.
      summary_writer: The SummaryWriter used to write summaries, or None if
        summaries are not used.
      profiler: The tf.profiler.Profiler, or None if tfprof is not used.

    Returns:
      Dictionary containing training statistics (num_workers, num_steps,
      average_wall_time, images_per_sec).
    """
    if bcast_global_variables_op:
      sess.run(bcast_global_variables_op)
    image_producer = None
    if graph_info.input_producer_op is not None:
      image_producer = cnn_util.ImageProducer(
          sess, graph_info.input_producer_op, self.batch_group_size,
          False)
      image_producer.start()
    if graph_info.enqueue_ops:
      for i in xrange(len(graph_info.enqueue_ops)):
        sess.run(graph_info.enqueue_ops[:(i + 1)])
        if image_producer is not None:
          image_producer.notify_image_consumption()
    self.init_global_step, = sess.run([graph_info.global_step])
    if self.job_name and not self.params.cross_replica_sync:
      # TODO(zhengxq): Do we need to use a global step watcher at all?
      global_step_watcher = GlobalStepWatcher(
          sess, graph_info.global_step,
          self.num_workers * self.num_warmup_batches +
          self.init_global_step,
          self.num_workers * (self.num_warmup_batches + self.num_batches) - 1)
      global_step_watcher.start()
    else:
      global_step_watcher = None
    eval_image_producer = None
    step_train_times = []
    log_fn('Running warm up')
    local_step = -1 * self.num_warmup_batches
    if self.single_session:
      # In single session mode, each step, the global_step is incremented by
      # 1. In non-single session mode, each step, the global_step is
      # incremented once per worker. This means we need to divide
      # init_global_step by num_workers only in non-single session mode.
      end_local_step = self.num_batches - self.init_global_step
    else:
      end_local_step = self.num_batches - (self.init_global_step //
                                           self.num_workers)
    if not global_step_watcher:
      # In cross-replica sync mode, all workers must run the same number of
      # local steps, or else the workers running the extra step will block.
      done_fn = lambda: local_step >= end_local_step
    else:
      done_fn = global_step_watcher.done

    skip_final_eval = False
    accuracy_at_1 = None
    last_eval_step = local_step
    loop_start_time = time.time()
    last_average_loss = None
    while not done_fn():
      if local_step == 0:
        log_fn('Done warm up')
        if graph_info.execution_barrier:
          log_fn('Waiting for other replicas to finish warm up')
          sess.run([graph_info.execution_barrier])

        # TODO(laigd): rename 'Img' to maybe 'Input'.
        header_str = ('Step\tImg/sec\t' +
                      self.params.loss_type_to_report.replace('/', ' '))
        if self.params.print_training_accuracy:
          # TODO(laigd): use the actual accuracy op names of the model.
          header_str += '\ttop_1_accuracy\ttop_5_accuracy'
        log_fn(header_str)
        assert len(step_train_times) == self.num_warmup_batches
        # reset times to ignore warm up batch
        step_train_times = []
        loop_start_time = time.time()
      if (summary_writer and
          (local_step + 1) % self.params.save_summaries_steps == 0):
        fetch_summary = graph_info.summary_op
      else:
        fetch_summary = None
      collective_graph_key = 7 if (
          self.params.variable_update == 'collective_all_reduce') else 0
      (summary_str, last_average_loss) = benchmark_one_step(
          sess, graph_info.fetches, local_step,
          self.batch_size * (self.num_workers
                             if self.single_session else 1), step_train_times,
          self.trace_filename, profiler, image_producer, self.params, 
          fetch_summary, collective_graph_key=collective_graph_key)
      if summary_str is not None and is_chief:
        supervisor.summary_computed(sess, summary_str)
      local_step += 1
      if (self.params.save_model_steps and
          local_step % self.params.save_model_steps == 0 and
          local_step > 0 and
          is_chief):
        supervisor.saver.save(sess, supervisor.save_path,
                              supervisor.global_step)
    loop_end_time = time.time()
    # Waits for the global step to be done, regardless of done_fn.
    if global_step_watcher:
      while not global_step_watcher.done():
        time.sleep(.25)
    if not global_step_watcher:
      elapsed_time = loop_end_time - loop_start_time
      average_wall_time = elapsed_time / local_step if local_step > 0 else 0
      images_per_sec = (self.num_workers * local_step * self.batch_size /
                        elapsed_time)
      num_steps = local_step * self.num_workers
    else:
      # NOTE: Each worker independently increases the global step. So,
      # num_steps will be the sum of the local_steps from each worker.
      num_steps = global_step_watcher.num_steps()
      elapsed_time = global_step_watcher.elapsed_time()
      average_wall_time = (elapsed_time * self.num_workers / num_steps
                           if num_steps > 0 else 0)
      images_per_sec = num_steps * self.batch_size / elapsed_time

    # We skip printing images/sec if --eval_during_training_* is specified,
    # because we are both processing training and evaluation images, so a
    # singular "images/sec" value is meaningless.
    log_fn('-' * 64)
    # TODO(laigd): rename 'images' to maybe 'inputs'.
    log_fn('total images/sec: %.2f' % images_per_sec)
    log_fn('-' * 64)
    num_steps_since_last_eval = local_step - last_eval_step
    python_global_step = sess.run(graph_info.global_step)
    num_epochs_ran = (python_global_step * self.batch_size /
                      self.dataset.num_examples_per_epoch('train'))
    if image_producer is not None:
      image_producer.done()
    if eval_image_producer is not None:
      eval_image_producer.done()
    if is_chief:
      log_fn('average_examples_per_sec: '
        '{} at global step {}'.format(images_per_sec, num_steps))

    # Save the model checkpoint.
    if self.params.train_dir is not None and is_chief:
      checkpoint_path = os.path.join(self.params.train_dir, 'model.ckpt')
      if not gfile.Exists(self.params.train_dir):
        gfile.MakeDirs(self.params.train_dir)
      supervisor.saver.save(sess, checkpoint_path, graph_info.global_step)
    if graph_info.execution_barrier:
      # Wait for other workers to reach the end, so this worker doesn't
      # go away underneath them.
      sess.run([graph_info.execution_barrier])
    stats = {
        'num_workers': self.num_workers,
        'num_steps': num_steps,
        'average_wall_time': average_wall_time,
        'images_per_sec': images_per_sec
    }
    if last_average_loss is not None:
      stats['last_average_loss'] = last_average_loss
    success = bool(self.model.reached_target())
    return stats

  def _build_input_processing(self, shift_ratio=0):
    """"Build the image (pre)processing portion of the model graph.

    Args:
      shift_ratio: shift_ratio for data_flow_ops.RecordInput.

    Returns:
      An InputProcessingInfo containing all the input sources to the model.
    """
    input_processing_info = InputProcessingInfo(
        input_producer_op=None,
        input_producer_stages=None,
        multi_device_iterator_input=None)

    # If using synthetic gpu inputs, do nothing on the cpu side.
    if self.dataset.use_synthetic_gpu_inputs():
      assert not self.datasets_use_prefetch
      return input_processing_info

    input_preprocessor = self.input_preprocessor

    # Use prefetching mechanism provided by dataset input pipeline.
    if self.datasets_use_prefetch:
      multi_device_iterator = (
          input_preprocessor.build_multi_device_iterator(
              self.batch_size, len(self.devices), self.cpu_device, self.params,
              self.raw_devices, self.dataset, False))
      return input_processing_info._replace(
          multi_device_iterator_input=multi_device_iterator.get_next())

    # Not using dataset prefetching. Use a staging area to mimic the prefetching
    # behavior instead.
    with tf.device(self.cpu_device):
      subset = 'train'
      input_list = input_preprocessor.minibatch(
          self.dataset,
          subset=subset,
          params=self.params,
          shift_ratio=shift_ratio)

      input_producer_op = []
      input_producer_stages = []
      for device_num in range(len(self.devices)):
        staging_area = data_flow_ops.StagingArea(
            [parts[0].dtype for parts in input_list],
            shapes=[parts[0].get_shape() for parts in input_list],
            shared_name='input_producer_staging_area_%d_eval_%s' %
            (device_num, False))
        input_producer_stages.append(staging_area)
        for group_index in xrange(self.batch_group_size):
          batch_index = group_index + device_num * self.batch_group_size
          put_op = staging_area.put(
              [parts[batch_index] for parts in input_list])
          input_producer_op.append(put_op)
      assert input_producer_op

    return input_processing_info._replace(
        input_producer_op=input_producer_op,
        input_producer_stages=input_producer_stages)

  def _maybe_initialize_fp16(self):
    """Initialize fp16 settings."""
    if self.params.use_fp16:
      init_loss_scale_val = float(self.params.fp16_loss_scale or
                                  self.model.get_fp16_loss_scale())
      self.loss_scale = None
      self.loss_scale_normal_steps = None
      if self.enable_auto_loss_scale or init_loss_scale_val != 1:
        self.loss_scale = tf.get_variable(
            name='loss_scale',
            initializer=init_loss_scale_val,
            dtype=tf.float32,
            trainable=False)
      if self.enable_auto_loss_scale:
        self.loss_scale_normal_steps = tf.get_variable(
            name='loss_scale_normal_steps', initializer=0, trainable=False)

  def _build_model(self):
    """Build the TensorFlow graph."""
    if self.datasets_use_prefetch:
      assert not self.variable_mgr.supports_staged_vars()

    # Adjust seed so different workers start read different input files.
    if self.params.variable_update == 'horovod':
      import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
      seed_adjustment = hvd.rank()
    else:
      seed_adjustment = 0
    tf.set_random_seed(1234 + seed_adjustment)
    np.random.seed(4321 + seed_adjustment)
    phase_train = True

    mode_string = 'training'

    log_fn('Generating {} model'.format(mode_string))
    losses = []
    device_grads = []
    all_logits = []
    all_accuracy_ops = {}
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()
      self._maybe_initialize_fp16()

    # Build the processing and model for the worker.
    input_producer_op = None
    with tf.name_scope('input_processing'):
      input_processing_info = self._build_input_processing(shift_ratio=0)
      if input_processing_info.input_producer_op is not None:
        input_producer_op = tf.group(*input_processing_info.input_producer_op)
    update_ops = None
    staging_delta_ops = []

    for device_num in range(len(self.devices)):
      with tf.name_scope('tower_%i' % device_num) as name_scope, (
          self.variable_mgr.create_outer_variable_scope(device_num)):
        results = self.add_forward_pass_and_gradients(
            phase_train, device_num, device_num, input_processing_info,
            gpu_compute_stage_ops, gpu_grad_stage_ops)

        if phase_train:
          losses.append(results['loss'])
          device_grads.append(results['gradvars'])
        else:
          all_logits.append(results['logits'])
        if not phase_train or self.params.print_training_accuracy:
          for name, op in results.items():
            if name.startswith('accuracy:'):
              key = name[9:]
              if key not in all_accuracy_ops:
                all_accuracy_ops[key] = []
              all_accuracy_ops[key].append(op)

        if device_num == 0:
          # Retain the Batch Normalization updates operations only from the
          # first tower. These operations update the moving mean and moving
          # variance variables, which are updated (but not used) during
          # training, and used during evaluation. The moving mean and variance
          # approximate the true mean and variance across all images in the
          # dataset. Therefore, in replicated mode, these moving averages would
          # be almost identical for each tower, and so we only update and save
          # the moving averages for one tower. In parameter server mode, all
          # towers share a copy of the variables so we also only need to update
          # and save the moving averages once.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
          if self.datasets_use_prefetch:
            assert not self.variable_mgr.staging_delta_ops
          else:
            staging_delta_ops = list(self.variable_mgr.staging_delta_ops)

    enqueue_ops = []
    if not self.datasets_use_prefetch:
      if self.variable_mgr.supports_staged_vars():
        for staging_ops in self.variable_mgr.staging_vars_on_devices:
          gpu_compute_stage_ops.extend(
              [put_op for _, (put_op, _) in six.iteritems(staging_ops)])
      enqueue_ops.append(tf.group(*gpu_compute_stage_ops,
                                  name='gpu_compute_stage_ops_group'))
      if gpu_grad_stage_ops:
        staging_delta_ops += gpu_grad_stage_ops
      if staging_delta_ops:
        enqueue_ops.append(tf.group(*(staging_delta_ops)))

    fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                  enqueue_ops, update_ops, all_accuracy_ops,
                                  phase_train)
    return (input_producer_op, enqueue_ops, fetches)

  def _build_fetches(self, global_step, all_logits, losses, device_grads,
                     enqueue_ops, update_ops, all_accuracy_ops, phase_train):
    """Complete construction of model graph, populating the fetches map."""
    fetches = {}
    if enqueue_ops:
      fetches['enqueue_ops'] = enqueue_ops
    for name, ops in all_accuracy_ops.items():
      # For fetches that starts with 'tensor:', keep dimension and skip reducing
      # them to scalars.
      if name.startswith(constants.UNREDUCED_ACCURACY_OP_PREFIX):
        key = name[len(constants.UNREDUCED_ACCURACY_OP_PREFIX):]
        fetches[key] = tf.concat(ops, 0)
      else:
        fetches[name] = tf.reduce_sum(ops) / self.batch_size
        if self.task_index == 0 and self.params.summary_verbosity >= 1:
          tf.summary.scalar(name, fetches[name])

    if not phase_train:
      return fetches
    apply_gradient_devices, gradient_state = (
        self.variable_mgr.preprocess_device_grads(device_grads))

    # TODO(reedwm): Greatly simplify the learning rate code.
    if (self.params.variable_update == 'horovod' or
        self.params.variable_update == 'collective_all_reduce'):
      # Each worker independently increments global_step.
      examples_per_step = self.batch_size * self.num_workers
    else:
      # global_step is shared by all workers, and so every iteration
      # global_step is incremented by num_workers.
      examples_per_step = self.batch_size

    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        with tf.name_scope('average_loss'):
          average_loss = tf.reduce_mean(losses)
        with tf.name_scope('get_gradients_to_apply'):
          avg_grads = self.variable_mgr.get_gradients_to_apply(d,
                                                               gradient_state)

        # We compute the learning rate once for each device in
        # `apply_gradient_devices`.
        learning_rate = get_learning_rate(
            self.params, global_step, self.dataset.num_examples_per_epoch(),
            self.model, examples_per_step)
        gradient_clip = self.params.gradient_clip
        if gradient_clip is not None:
          with tf.name_scope('clip_gradients'):
            clipped_grads = [(tf.clip_by_value(grad, -gradient_clip,
                                               +gradient_clip), var)
                             for grad, var in avg_grads]
        else:
          clipped_grads = avg_grads

        learning_rate = tf.identity(learning_rate, name='learning_rate_tensor')
        opt = get_optimizer(self.params, learning_rate)
        loss_scale_params = variable_mgr_util.AutoLossScaleParams(
            enable_auto_loss_scale=self.enable_auto_loss_scale,
            loss_scale=self.loss_scale,
            loss_scale_normal_steps=self.loss_scale_normal_steps,
            inc_loss_scale_every_n=self.params.fp16_inc_loss_scale_every_n,
            is_chief=not self.job_name or self.task_index == 0)

        with tf.name_scope('append_apply_gradient_ops'):
          self.variable_mgr.append_apply_gradients_ops(
              gradient_state, opt, clipped_grads, training_ops,
              loss_scale_params)
    train_op = tf.group(*(training_ops + update_ops), name='train_ops_group')

    with tf.device(self.cpu_device):
      if self.task_index == 0 and self.params.summary_verbosity >= 1:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar(self.params.loss_type_to_report, average_loss)
        if self.loss_scale is not None:
          tf.summary.scalar('loss_scale', self.loss_scale)
        if self.loss_scale_normal_steps:
          tf.summary.scalar('loss_scale_normal_steps',
                            self.loss_scale_normal_steps)

        if self.params.summary_verbosity >= 2:
          self.gradient_histogram_summary(avg_grads)

        if self.params.summary_verbosity >= 3:
          for grad, var in avg_grads:
            if grad is not None:
              tf.summary.histogram(var.op.name + '/gradients', grad)
          for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    fetches['train_op'] = train_op
    fetches['average_loss'] = average_loss
    return fetches

  def gradient_histogram_summary(self, avg_grads):
    """Create histogram of log values of all non-zero gradients."""
    with tf.name_scope('log_gradients_summary'):
      all_grads = []
      for grad, _ in avg_grads:
        all_grads.append(tf.reshape(grad, [-1]))
      grads = tf.abs(tf.concat(all_grads, 0))
      # exclude grads with zero values.
      indices_for_non_zero_grads = tf.where(tf.not_equal(grads, 0))
      log_grads = tf.reshape(
          tf.log(tf.gather(grads, indices_for_non_zero_grads)), [-1])
      tf.summary.histogram('log_gradients', log_grads)

  def _build_model_single_session(self):
    """Build the TensorFlow graph for multiple replicas in a single_session.

    Returns:
      input_producer_op:
      enqueue_ops:
      fetches:

    Raises:
       ValueError: optimizer not recognized.

    Single session runs multiple model replicas as part of one large
    distributed graph, whose global execution is always step-synchronized.
    """
    # verify assumptions
    assert self.params.task_index == 0

    tf.set_random_seed(1234)
    np.random.seed(4321)
    phase_train = True

    log_fn('Generating training model')
    losses = []
    device_grads = []
    all_logits = []
    all_accuracy_ops = {}
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.train.get_or_create_global_step()

    update_ops = []
    global_input_producer_op = []

    is_local = not self.job_name
    if is_local:
      assert self.num_workers == 1
    for task_num in range(self.num_workers):
      # Reset the devices that self.variable_mgr knows about to those
      # belonging to the next worker (task).
      self.reset_devices_for_task(task_num, is_local)
      # Build the per-worker image processing
      with tf.name_scope('input_processing'):
        input_processing_info = self._build_input_processing(
            shift_ratio=(task_num / self.num_workers))
      if input_processing_info.input_producer_op is not None:
        global_input_producer_op.extend(input_processing_info.input_producer_op)
      # Build the per-worker model replica.
      for rel_device_num in range(len(self.devices)):
        abs_device_num = task_num * len(self.devices) + rel_device_num
        with self.variable_mgr.create_outer_variable_scope(
            abs_device_num), tf.name_scope(
                'task_%i_tower_%i' % (task_num, rel_device_num)) as name_scope:
          task_results = self.add_forward_pass_and_gradients(
              phase_train, rel_device_num, abs_device_num,
              input_processing_info, gpu_compute_stage_ops, gpu_grad_stage_ops)

          if phase_train:
            losses.append(task_results['loss'])
            device_grads.append(task_results['gradvars'])
          else:
            all_logits.append(task_results['logits'])
          if not phase_train or self.params.print_training_accuracy:
            for name, op in task_results.items():
              if name.startswith('accuracy:'):
                key = name[9:]
                if key not in all_accuracy_ops:
                  all_accuracy_ops[key] = []
                all_accuracy_ops[key].append(op)

          if rel_device_num == 0:
            # Retain the Batch Normalization updates operations only
            # from the first tower. These operations update the moving
            # mean and moving variance variables, which are updated
            # (but not used) during training, and used during
            # evaluation. The moving mean and variance approximate the
            # true mean and variance across all images in the
            # dataset. Therefore, in replicated mode, these moving
            # averages would be almost identical for each tower, and
            # so we only update and save the moving averages for one
            # tower. In parameter server mode, all towers share a copy
            # of the variables so we also only need to update and save
            # the moving averages once.
            update_ops.extend(
                tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope))
            assert not self.variable_mgr.staging_delta_ops

    enqueue_ops = []
    if gpu_compute_stage_ops:
      enqueue_ops.append(tf.group(*gpu_compute_stage_ops,
                                  name='gpu_compute_stage_ops'))
    assert not self.variable_mgr.supports_staged_vars()
    assert not gpu_grad_stage_ops

    fetches = self._build_fetches(global_step, all_logits, losses, device_grads,
                                  enqueue_ops, update_ops, all_accuracy_ops,
                                  phase_train)
    if global_input_producer_op:
      global_input_producer_op = tf.group(*global_input_producer_op)
    else:
      global_input_producer_op = None
    return (global_input_producer_op, enqueue_ops, fetches)

  def add_forward_pass_and_gradients(self,
                                     phase_train,
                                     rel_device_num,
                                     abs_device_num,
                                     input_processing_info,
                                     gpu_compute_stage_ops,
                                     gpu_grad_stage_ops):
    """Add ops for forward-pass and gradient computations."""
    nclass = self.dataset.num_classes
    if self.datasets_use_prefetch:
      assert input_processing_info.multi_device_iterator_input, (
          'multi_device_iterator_input cannot be None if '
          'datasets_use_prefetch=True')
      input_list = (
          input_processing_info.multi_device_iterator_input[rel_device_num])
    else:
      if not self.dataset.use_synthetic_gpu_inputs():
        input_producer_stage = input_processing_info.input_producer_stages[
            rel_device_num]
        with tf.device(self.cpu_device):
          host_input_list = input_producer_stage.get()
        with tf.device(self.raw_devices[rel_device_num]):
          gpu_compute_stage = data_flow_ops.StagingArea(
              [inp.dtype for inp in host_input_list],
              shapes=[inp.get_shape() for inp in host_input_list])
          # The CPU-to-GPU copy is triggered here.
          gpu_compute_stage_op = gpu_compute_stage.put(host_input_list)
          input_list = gpu_compute_stage.get()
          gpu_compute_stage_ops.append(gpu_compute_stage_op)
      else:
        with tf.device(self.raw_devices[rel_device_num]):
          # Minor hack to avoid H2D copy when using synthetic data
          input_list = self.model.get_synthetic_inputs(
              BenchmarkCNN.GPU_CACHED_INPUT_VARIABLE_NAME, nclass)

    def device_aware_reshape(tensor, shape):
      device = self.devices[rel_device_num]
      # Labels are int32, place reshapes on gpu:0 (no device placement) when the
      # hack is enabled.
      with tf.device(device):
        return tf.reshape(tensor, shape=shape)

    subset = 'train'
    input_shapes = self.model.get_input_shapes(subset)
    input_list = [
        device_aware_reshape(input_list[i], shape=input_shapes[i])
        for i in range(len(input_list))
    ]

    def forward_pass_and_gradients():
      """Builds forward pass and gradient computation network.

      When phase_train=True and print_training_accuracy=False:
        return [loss] + grads

      When phase_train=True and print_training_accuracy=True:
        return [logits, loss] + grads

      When phase_train=False,
        return [logits]

      Its output can always be unpacked by

      ```
        outputs = forward_pass_and_gradients()
        logits, loss, grads = unpack_forward_pass_and_gradients_output(outputs)
      ```

      Returns:
        outputs: A list of tensors depending on different modes.
      """

      build_network_result = self.model.build_network(
          input_list, phase_train, nclass)
      logits = build_network_result.logits

      if not phase_train:
        return [logits]

      base_loss = self.model.loss_function(input_list, build_network_result)
      params = self.variable_mgr.trainable_variables_on_device(
          rel_device_num, abs_device_num)
      l2_loss = None
      total_loss = base_loss
      with tf.name_scope('l2_loss'):
        fp32_params = params
        if self.model.data_type == tf.float16 and self.params.fp16_vars:
          # fp16 reductions are very slow on GPUs, so cast to fp32 before
          # calling tf.nn.l2_loss and tf.add_n.
          # TODO(b/36217816): Once the bug is fixed, investigate if we should do
          # this reduction in fp16.
          fp32_params = (tf.cast(p, tf.float32) for p in params)
        filtered_params = self.model.filter_l2_loss_vars(fp32_params)
        if rel_device_num == len(self.devices) - 1:
          # We compute the L2 loss for only one device instead of all of them,
          # because the L2 loss for each device is the same. To adjust for this,
          # we multiply the L2 loss by the number of devices. We choose the
          # last device because for some reason, on a Volta DGX1, the first four
          # GPUs take slightly longer to complete a step than the last four.
          # TODO(reedwm): Shard the L2 loss computations across GPUs.
          if self.params.single_l2_loss_op:
            # TODO(reedwm): If faster, create a fused op that does the L2 loss
            # on multiple tensors, and use that instead of concatenating
            # tensors.
            reshaped_params = [tf.reshape(p, (-1,)) for p in filtered_params]
            l2_loss = tf.nn.l2_loss(tf.concat(reshaped_params, axis=0))
          else:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in filtered_params])
      weight_decay = self.params.weight_decay
      if (weight_decay is not None and weight_decay != 0. and
          l2_loss is not None):
        total_loss += len(self.devices) * weight_decay * l2_loss

      aggmeth = tf.AggregationMethod.DEFAULT
      scaled_loss = (total_loss if self.loss_scale is None
                     else total_loss * self.loss_scale)
      grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
      if self.loss_scale is not None:
        # TODO(reedwm): If automatic loss scaling is not used, we could avoid
        # these multiplications by directly modifying the learning rate instead.
        # If this is done, care must be taken to ensure that this scaling method
        # is correct, as some optimizers square gradients and do other
        # operations which might not be compatible with modifying both the
        # gradients and the learning rate.

        grads = [
            grad * tf.cast(1. / self.loss_scale, grad.dtype) for grad in grads
        ]

      if self.params.variable_update == 'horovod':
        import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
        if self.params.horovod_device:
          horovod_device = '/%s:0' % self.params.horovod_device
        else:
          horovod_device = ''
        # All-reduce gradients using Horovod.
        grads = [hvd.allreduce(grad, average=False, device_dense=horovod_device)
                 for grad in grads]

      if self.params.loss_type_to_report == 'total_loss':
        loss = total_loss
      else:
        loss = base_loss

      if self.params.print_training_accuracy:
        return [logits, loss] + grads
      else:
        return [loss] + grads

    def unpack_forward_pass_and_gradients_output(forward_pass_and_grad_outputs):
      """Unpacks outputs from forward_pass_and_gradients.

      Args:
        forward_pass_and_grad_outputs: Output from forward_pass_and_gradients.

      Returns:
        logits: Unscaled probability distribution from forward pass.
          If unavailable, None is returned.
        loss: Loss function result from logits.
          If unavailable, None is returned.
        grads: Gradients for all trainable variables.
          If unavailable, None is returned.
      """
      logits = None
      # logits is only fetched in non-train mode or when
      # print_training_accuracy is set.
      if not phase_train or self.params.print_training_accuracy:
        logits = forward_pass_and_grad_outputs.pop(0)

      loss = (
          forward_pass_and_grad_outputs[0]
          if forward_pass_and_grad_outputs else None)
      grads = (
          forward_pass_and_grad_outputs[1:]
          if forward_pass_and_grad_outputs else None)

      return logits, loss, grads

    def make_results(logits, loss, grads):
      """Generate results based on logits, loss and grads."""
      results = {}  # The return value

      if logits is not None:
        results['logits'] = logits
        accuracy_ops = self.model.accuracy_function(input_list, logits)
        for name, op in accuracy_ops.items():
          results['accuracy:' + name] = op

      if loss is not None:
        results['loss'] = loss

      if grads is not None:
        param_refs = self.variable_mgr.trainable_variables_on_device(
            rel_device_num, abs_device_num, writable=True)
        results['gradvars'] = list(zip(grads, param_refs))

      return results

    with tf.device(self.devices[rel_device_num]):
      outputs = forward_pass_and_gradients()
      logits, loss, grads = unpack_forward_pass_and_gradients_output(outputs)
      return make_results(logits, loss, grads)

  def get_input_preprocessor(self):
    """Returns the image preprocessor to used, based on the model.

    Returns:
      The image preprocessor, or None if synthetic data should be used.
    """
    shift_ratio = 0
    if self.job_name:
      # shift_ratio prevents multiple workers from processing the same batch
      # during a step
      shift_ratio = self.task_index / self.num_workers

    processor_class = self.dataset.get_input_preprocessor(
        self.params.input_preprocessor)
    assert processor_class
    subset = 'train'
    return processor_class(
        self.batch_size * self.batch_group_size,
        self.model.get_input_shapes(subset),
        len(self.devices) * self.batch_group_size,
        dtype=self.model.data_type,
        train=True,
        # TODO(laigd): refactor away image model specific parameters.
        distortions=self.params.distortions,
        resize_method=self.resize_method,
        shift_ratio=shift_ratio,
        summary_verbosity=self.params.summary_verbosity,
        distort_color_in_yiq=True,
        fuse_decode_and_crop=True,
        match_mlperf=None)

  def add_sync_queues_and_barrier(self, name_prefix, enqueue_after_list):
    """Adds ops to enqueue on all worker queues.

    Args:
      name_prefix: prefixed for the shared_name of ops.
      enqueue_after_list: control dependency from ops.

    Returns:
      An op that should be used as control dependency before starting next step.
    """
    self.sync_queue_counter += 1
    with tf.device(self.sync_queue_devices[(
        self.sync_queue_counter % len(self.sync_queue_devices))]):
      sync_queues = [
          tf.FIFOQueue(self.num_workers, [tf.bool], shapes=[[]],
                       shared_name='%s%s' % (name_prefix, i))
          for i in range(self.num_workers)]
      queue_ops = []
      # For each other worker, add an entry in a queue, signaling that it can
      # finish this step.
      token = tf.constant(False)
      with tf.control_dependencies(enqueue_after_list):
        for i, q in enumerate(sync_queues):
          if i == self.task_index:
            queue_ops.append(tf.no_op())
          else:
            queue_ops.append(q.enqueue(token))

      # Drain tokens off queue for this worker, one for each other worker.
      queue_ops.append(
          sync_queues[self.task_index].dequeue_many(len(sync_queues) - 1))

      return tf.group(*queue_ops)

def _set_environ_vars(params):
  """Sets up the environment variables that BenchmarkCNN should use."""
  os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  os.environ['TF_SYNC_ON_FINISH'] = '0'
  argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Sets GPU thread settings
  if params.device.lower() == 'gpu':
    params = params._replace(gpu_thread_mode=params.gpu_thread_mode.lower())
    if params.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
      raise ValueError('Invalid gpu_thread_mode: %s' % params.gpu_thread_mode)
    os.environ['TF_GPU_THREAD_MODE'] = params.gpu_thread_mode

    if params.per_gpu_thread_count and params.gpu_thread_mode == 'global':
      raise ValueError(
          'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
          params.per_gpu_thread_count)
    # Default to two threads. One for the device compute and the other for
    # memory copies.
    per_gpu_thread_count = params.per_gpu_thread_count or 2
    total_gpu_thread_count = per_gpu_thread_count * params.num_gpus

    if params.gpu_thread_mode == 'gpu_private':
      os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
    elif params.gpu_thread_mode == 'gpu_shared':
      os.environ['TF_GPU_THREAD_COUNT'] = str(total_gpu_thread_count)

    cpu_count = multiprocessing.cpu_count()

    if (params.datasets_use_prefetch and
        params.datasets_num_private_threads is None):
      # From the total cpu thread count, subtract the total_gpu_thread_count,
      # and then 2 threads per GPU device for event monitoring and sending /
      # receiving tensors
      num_monitoring_threads = 2 * params.num_gpus
      num_private_threads = max(
          cpu_count - total_gpu_thread_count - num_monitoring_threads, 1)
      params = params._replace(datasets_num_private_threads=num_private_threads)
  return params


def setup(params):
  """Sets up the environment that BenchmarkCNN should run in.

  Args:
    params: Params tuple, typically created by make_params_from_flags.

  Returns:
    A potentially modified params.
  Raises:
    ValueError: invalid parames combinations.
  """
  # Set up environment variables before doing any other global initialization to
  # make sure it uses the appropriate environment variables.
  params = _set_environ_vars(params)

  # horovod needs to be initialized before create_config_proto() call since
  # it will be used in config generation if enabled.
  if params.variable_update == 'horovod':
    import horovod.tensorflow as hvd  # pylint: disable=g-import-not-at-top
    hvd.init()

  if not params.job_name:
    # Create a dummy session to initialize TF global variables using the input
    # params. Otherwise, ListDevices function may create global devices using
    # the default config instead of using the user provided config.
    #
    # TODO(hinsu): Find a way to achieve the same for distributed benchmark. It
    # is not legal to create distributed session after local session. It is also
    # not possible to create distributed session here as that results in
    # multiple creation of ClusterManager and Server.
    with tf.Session(config=create_config_proto(params)) as sess:
      del sess

  return params
