"""Benchmark script for TensorFlow"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def main(positional_arguments):
    assert len(positional_arguments) >= 1
    if len(positional_arguments) > 1:
        raise ValueError('Received unknow positional argument: %s'
                         % positional_arguments[1:])

if __name__ == "__main__":
    main()
