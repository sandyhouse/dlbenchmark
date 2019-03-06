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

"""Benchmark script for PyTorch."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

try:
  import torch
except ImportError:
  print("To run benchmarks for PyTorch backend, PyTorch should be installed.")
  print("Instructions for install PyTorch are as follow:")
  print("  - https://pytorch.org")
  raise

import pytorch.utils as utils
import pytorch.benchmark_cnn as benchmark_cnn
# All supported NLP models to benchmark.
NLP_MODELS = ['bert', 'transformer']

def main(params):
  """
  Run benchmarks for PyTorch.

  Args:
    params: commline line arguments.
  """
  print("Run benchmarks for PyTorch.")
  version = utils.get_torch_version()
  print("PyTorch:     %i.%i" % (version[0], version[1]))


  if params.model in NLP_MODELS:
    raise NotImplemented("The benchmarks for PyTorch backend have not been "
                         "implemented.")
    bench = benchmark_nlp.BenchmarkNLP(params)
  else:
    bench = benchmark_cnn.BenchmarkCNN(params)
  
  bench.run()
