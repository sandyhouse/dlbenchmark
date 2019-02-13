# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf


def get_learning_rate(init_lr, hidden_size, learning_rate_warmup_steps):
  """Calculate learning rate with linear warmup and rsqrt decay."""
  learning_rate = init_lr
  with tf.name_scope("learning_rate"):
    warmup_steps = None
    if learning_rate_warmup_steps:
      warmup_steps = tf.to_float(learning_rate_warmup_steps)
    step = tf.to_float(tf.train.get_or_create_global_step())

    learning_rate *= (hidden_size ** -0.5)
    if warmup_steps:
      # Apply linear warmup
      learning_rate *= tf.minimum(1.0, step / warmup_steps)
    if warmup_steps == None:
      warmup_steps = 0

    # Apply rsqrt decay
    learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

    tf.identity(learning_rate, "learning_rate")

    return learning_rate

def create_optimizer(loss, init_lr, hidden_size, num_train_steps, 
                     num_warmup_steps):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()
  learning_rate = get_learning_rate(init_lr, hidden_size, num_warmup_steps)

  optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  return train_op
