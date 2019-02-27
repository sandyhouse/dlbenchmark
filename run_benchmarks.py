# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

"""Benchmark scripts for TensorFlow and PyTorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import flags

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
    print("The model to benchmark is not specified.\n"
          "Using `--model` to specify a model to benchmark.")
    print("All supported models are as follows:")
    print("=" * 30)
    for model in ALL_MODELS:
      print("    -%s" % model)
    print("=" * 30)
    print("Please specify one of the above models.")
    exit()

  if not params.model in ALL_MODELS:
    print("The model `%s` is not implemented in our benchmarks." % params.model)
    print("All supported models are as follows:")
    print("=" * 30)
    for model in ALL_MODELS:
      print("    -%s" % model)
    print("=" * 30)
    print("Please specify one of the above models.")
    exit()
  
  if not (params.do_train or params.do_eval or params.do_predict):
    print("At least one of `--do_train`, `--do_eval` or `--do_predict` "
          "must be specified.")
    exit()
  
  if (params.num_gpus == 0 and params.data_format != 'NHWC') or (
      params.num_gpus and params.data_format != 'NCHW'):
    print("Required: 'NHWC' for CPU and 'NCHW' for GPU")
    exit()

  if params.ip_list:
    if params.job_name == None or params.job_index == None:
      print("`--job_name` and `--job_index` must both be specified for "
            "distributed training/evaluation/prediction.")
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

  if params.backend in ['TensorFlow', 'tensorflow']:
    import tf_benchmarks
    tf_benchmarks.main(params)
  else:
    import torch.torch_benchmarks as torch_benchmarks
    torch_benchmarks.main(params)

if __name__ == '__main__':
  main(params)
