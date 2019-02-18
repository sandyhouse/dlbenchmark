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

"""Benchmark script for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

import flags
import tf_benchmarks
import torch.torch_benchmarks

# All supported models to benchmark.
ALL_MODELS = ['alexnet',
              'resnet50',
              'googlenet',
              'vgg16',
              'bert',
              'transformer',
             ]

# All supported NLP models to benchmark.
NLP_MODELS = ['bert', 'transformer']

# Define and parse command line arguments.
flags.define_flags()
params = flags.parser.parse_args()

def _validate_flags(params):
  """Check if command line arguments are valid."""
  if params.model is None:
    print("The model to benchmark is not specified, using `--model` "
          "to specify a model to benchmark.")
    exit()

  if not params.model in ALL_MODELS:
    print("The model `%s` is not implemented in our benchmarks." % params.model)
    print("All supported models to benchmark:")
    print("=" * 30)
    for model in ALL_MODELS:
      print("*** {}".format(model))
    print("=" * 30)
    print("Please specify one of the above models.")
    exit()
  
  if not params.do_train and not params.do_eval:
    print("At least one of `--do_train` or `--do_eval` should be specified.")
    exit()
  
  if((params.num_epochs_per_decay or
      params.learning_rate_decay_factor) and
      not (params.init_learning_rate is not None and
           params.num_epochs_per_decay and 
           params.learning_rate_decay_factor)):
    print("If one of `--num_epochs_per_decay` or `-learning_rate_decay_factor` "
          "is set, both must be set and `--init_learning_rate` must be set.")
    exit()

  if (params.minimum_learning_rate and
      not (params.init_learning_rate is not None and
           params.num_epochs_per_decay and 
           params.learning_rate_decay_factor)):
    print("`--minimum_learning_rate` requires `--init_learning_rate`"
          ", `--num_epochs_per_decay`, and `--learning_rate_decay_factor` "
          "to be set.")
    exit()

  if params.fp16_vars and not params.use_fp16:
    print('If `--fp16_vars` is specified, set `--use_fp16` to True at the same '
          'time.')
    exit()

  if (params.use_fp16 and params.fp16_vars and
      params.all_reduce_spec and 'nccl' in params.all_reduce_spec):
    print('fp16 variables are not supported with NCCL.')
    exit()
  

  if (params.num_gpus == 0 and params.data_format != 'NHWC') or (
      params.num_gpus and params.data_format != 'NCHW'):
    print("Recommend: 'NHWC' for CPU and 'NCHW' for GPU")
    exit()

  if params.ip_list:
    if params.job_name == None or params.job_index == None:
      print("`--job_name` and `--job_index` should both be specified for "
            "distributed training/evaluation. ")
      exit()

def main(params):
  """Run the benchmark."""
  _validate_flags(params)

  if params.ip_list:
    ips = params.ip_list.split(',')

    TF_CONFIG = {}
    addresses = []
    if params.job_name == 'ps':
      port = '5000'
    else:
      port = '5001'

    for ip in ips:
      address = ip + ":" + port
      addresses.append(address)
    
    TF_CONFIG['cluster'] = {params.job_name : addresses}
    TF_CONFIG['task'] = {
            'type': params.job_name,
            'index': params.job_index,
    }

    os.environ["TF_CONFIG"] = json.dumps(TF_CONFIG)

  if params.backend == 'TensorFlow':
    tf_benchmarks.main(params)
  else:
    torch_benchmarks.main(params)

if __name__ == '__main__':
  main(params)
