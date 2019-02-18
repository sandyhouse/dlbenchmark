
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch

import utils
import benchmark_cnn

# All supported NLP models to benchmark.
NLP_MODELS = ['bert', 'transformer']

def main(params):
  """
  Run benchmarks for PyTorch.

  Args:
    params: commline line arguments.
  """
  version = utils.get_torch_version()

  print("PyTorch:     %i.%i" % (version[0], version[1]))


  if params.model in NLP_MODELS:
    raise NotImplemented("The benchmarks for PyTorch backend have not been "
                         "implemented.")
    bench = benchmark_nlp.BenchmarkNLP(params)
  else:
    bench = benchmark_cnn.BenchmarkCNN(params)
  
  bench.run()

if __name__ == "__main__":
  main(None)
