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
"""Alexnet model configuration.

References:
  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton
  ImageNet Classification with Deep Convolutional Neural Networks
  Advances in Neural Information Processing Systems. 2012
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn

default_config = {
    'image_size': 224,
    'num_classes': 1000
    }

class AlexnetModel(nn.Module):
  """Alexnet model for ImageNet."""

  def __init__(self, num_classes=default_config['num_classes']):
    super(AlexnetModel, self).__init__()
    self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            )
    self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            )

    

  def forward(self, inputs):
    """Builds the forward pass of the model.

    Args:
      inputs: the list of inputs, excluding labels
      is_training: if in the phrase of training.

    Returns:
      The logits of the model.
    """
    output = self.conv(inputs)
    output = output.view(x.size(0), 256 * 6 * 6)
    logits = self.classifier(output)
    return logits
