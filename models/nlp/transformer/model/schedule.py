# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Abstract training on a step or epoch basis."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf


_TRAIN, _EVAL = tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL


NUM_EXAMPLES = {
    tf.estimator.ModeKeys.TRAIN: 4572160,
    # # Examples that are too long are filtered out, thus the total is less
    # # than the total number of lines.
    # 2399123 +  # news-commentary-v12.de-en
    # 1920209 +  # commoncrawl.de-en
    # 270769,    # europarl-v7.de-en
    tf.estimator.ModeKeys.EVAL: 3000,  # newstest2013
}


class Manager(object):
  """Container for convenience functions to abstract step or epoch basis.
  Transformer allows users to specify an epoch basis (generally recommended for
  full training) or a number of steps basis (convenient since epochs are rather
  large). TPUs furthermore require a step basis; however epochs are the norm in
  the machine learning community and it is desirable to allow users to specify
  epochs even when running with TPUS which requires behind the scenes
  conversions.
  This container simply groups what are largely mundane checks and conversions
  rather than interspersing them throughout the run loop code.
  """

  def __init__(self, train_steps, steps_between_evals, train_epochs,
               epochs_between_evals, default_train_epochs, batch_size,
               max_length):
    if train_steps and train_epochs:
      raise ValueError("Both train_steps or train_epochs were be defined.")

    # Determine training schedule based on flags.
    if train_steps:
      self.train_eval_iterations = train_steps // steps_between_evals
      self._single_iteration_train_steps = steps_between_evals
      self._single_iteration_train_epochs = None
    else:
      train_epochs = train_epochs or default_train_epochs
      self.train_eval_iterations = train_epochs // epochs_between_evals
      self._single_iteration_train_steps = None
      self._single_iteration_train_epochs = epochs_between_evals

    self.max_length = max_length
    self.batch_size = batch_size

  @property
  def single_iteration_train_steps(self):
    return self._single_iteration_train_steps

  @property
  def single_iteration_eval_steps(self):
    return None

  @property
  def train_increment_str(self):
    if self._single_iteration_train_steps:
      return "{} steps.".format(self._single_iteration_train_steps)

    return "{} epochs.".format(self._single_iteration_train_epochs)

  @property
  def repeat_dataset(self):
    if (self._single_iteration_train_epochs is None and
        self._single_iteration_train_steps > NUM_EXAMPLES[_TRAIN]):
      return math.ceil(self._single_iteration_train_steps /
                       NUM_EXAMPLES[_TRAIN])
    return self._single_iteration_train_epochs
