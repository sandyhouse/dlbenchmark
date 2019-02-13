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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import multiprocessing

parser = argparse.ArgumentParser(
        usage='%(prog)s [options]',
        description="Benchmarks for Deep Learning Platform - TensorFlow.")

# Define commond line arguments for benchmarks
def define_flags():
  parser.add_argument("--model",
                      type=str,
                      default=None,
                      help="Name of the model to run.")
  parser.add_argument("--batch_size",
                      type=int,
                      default=32,
                      help="Batch size per computing device.")
  parser.add_argument("--num_epochs",
                      type=int,
                      default=1,
                      help="Number of epochs to run.")
  parser.add_argument("--num_gpus",
                      type=int,
                      default=1,
                      help="Number of GPUs to use. If the value is set to 0, "
                           "CPUs will be used instead.")
  parser.add_argument("--data_dir",
                      type=str,
                      default=None,
                      help="Directory for input dataset. If it is not set, "
                           "synthetic data will be used.")
  parser.add_argument("--do_train",
                      default=False,
                      action='store_true',
                      help="Whether to run training.")
  parser.add_argument("--do_eval",
                      default=False,
                      action='store_true',
                      help="Whether to run evaluation.")
  parser.add_argument("--data_format",
                      choices=['NHWC', 'NCHW'],
                      default="NCHW",
                      help="Data format to use. Recommended: NHWC for CPUs, "
                           "and NCHW for GPUs.")
  parser.add_argument("--tfprof_file",
                      type=str,
                      default=None,
                      help="File to write the tfprof ProfileProto. The "
                           "performance and other aspects of the model can be "
                           "analyzed with tfprof. See "
                           "https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/g3doc/command_line.md "
                           "for more information. Additionally, the top most "
                           "time consuming ops will be showed. Note: profiling "
                           "with tfprof is very slow, but most of the overhead "
                           "is spent between steps. So profiling results are "
                           "more accurate than the slowdown would suggest.")
  parser.add_argument("--optimizer",
                      choices=['momentum', 'sgd', 'rmsprop', 'adam'],
                      default="adam",
                      help="Optimizer to use.")
  parser.add_argument("--init_learning_rate",
                      type=float,
                      default=5e-5,
                      help="Initial learning rate to use.")
  parser.add_argument("--num_epochs_per_decay",
                      type=int,
                      default=0,
                      help="Epochs after which learning rate decays.")
  parser.add_argument("--learning_rate_decay_factor",
                      type=float,
                      default=0,
                      help="Learning rate decay factor every "
                           "`num_epochs_per_decay` epochs.")
  parser.add_argument("--minimum_learning_rate",
                      type=float,
                      default=0.0,
                      help="Minimum learning rate. Requires "
                           "`init_learning_rate`, `num_epochs_per_decay` and "
                           "`learning_rate_decay_factor` to be specified.")
  parser.add_argument("--momentum",
                      type=float,
                      default=0.9,
                      help="Momentum for training.")
  parser.add_argument("--rmsprop_decay",
                      type=float,
                      default=0.9,
                      help="Decay term for rmsprop.")
  parser.add_argument("--rmsprop_momentum",
                      type=float,
                      default=0.0,
                      help="Momentum in rmsprop.")
  parser.add_argument("--rmsprop_epsilon",
                      type=float,
                      default=1.0e-10,
                      help="Epsilon term for rmsprop.")
  parser.add_argument("--adam_beta1",
                      type=float,
                      default=0.9,
                      help="Beta1 term for Adam optimizer.")
  parser.add_argument("--adam_beta2",
                      type=float,
                      default=0.999,
                      help="Beta2 term for Adam optimizer.")
  parser.add_argument("--adam_epsilon",
                      type=float,
                      default=1e-8,
                      help="Epsilon term for Adam optimizer.")
  parser.add_argument("--use_fp16",
                      default=False,
                      action='store_true',
                      help="Whether to use float16 for tensors.")
  parser.add_argument("--fp16_vars",
                      default=False,
                      action='store_true',
                      help="Whether to use fp16 for variables. If False, "
                           "variables are stored in fp32 and casted to fp16 "
                           "when used. Recommended: False.")
  parser.add_argument("--all_reduce_spec",
                      type=str,
                      default=None,
                      help="A specification of the all_reduce algorithm to use "
                           "for reducing gradients.")
  parser.add_argument("--save_checkpoints_steps",
                      type=int,
                      default=None,
                      help="How often to save trained models.")
  parser.add_argument("--max_chkpts_to_keep",
                      type=int,
                      default=5,
                      help="Maximum number of checkpoints to keep.")
  parser.add_argument("--output_dir",
                      type=str,
                      default=None,
                      help="Path where model checkpoints will be written.")

# Flags for NLP models
  parser.add_argument("--vocab_file",
                     type=str,
                     default=None,
                     help="The vocabulary file to train on.")
  parser.add_argument("--init_checkpoint",
                     type=str,
                     default=None,
                     help="Initial checkpoint from a pre-trained model.")
  parser.add_argument("--max_seq_length",
                     type=int,
                     default=128,
                     help="Maximum length of input sequence. Sequences longer "
                          "than that will be truncated, and sequences shorter "
                          "than that will be padded.")
  parser.add_argument("--param_set",
                      type=str,
                      default="big",
                      help="Parameter set to use when creating and training "
                           "the model. The parameters define the input shape "
                           "(batch batch_size and max length), model "
                           "configuration (size of embedding, # of hidden "
                           "layers, etc.), and various other settings. "
                           "The big parameter set increases the default batch "
                           "size, embedding/hidden size, and filter size.")
  parser.add_argument("--num_parallel_calls",
                      type=int,
                      default=multiprocessing.cpu_count(),
                      help="The number of records that are processed in "
                           "parallel during input processing. This can be "
                           "optimized per data set but for generally "
                           "homogeneous data sets, should be approximately the "
                           "number of available CPU cores.")
  parser.add_argument("--use_synthetic_data",
                      type=bool,
                      default=False,
                      help="If set, use fake data (zeros) instead of a real "
                           "dataset.")
  parser.add_argument("--static_batch",
                      type=bool,
                      default=False,
                      help="Whether the batches in the dataset should have "
                           "static.")
  parser.add_argument("--bleu_source",
                      type=str,
                      default=None,
                      help="Path to source file containing text translate when "
                           "calculating the official BLEU score.")
  parser.add_argument("--bleu_ref",
                      type=str,
                      default=None,
                      help="Path to source file containing text translate when "
                           "calculating the official BLEU score.")

  parser.add_argument("--bert_config_file",
                     type=str,
                     default=None,
                     help="The config file (json) for the BERT model, which "
                          "specifies the model architecture.")
  parser.add_argument("--task_name",
                     type=str,
                     default=None,
                     help="The name of the task to train.")
  parser.add_argument("--do_lower_case",
                     type=bool,
                     default=True,
                     help="Whether to use low case for input text.")
