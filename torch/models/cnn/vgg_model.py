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

"""Resnet model configuration.

References:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition
  arXiv:1512.03385 (2015)

  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks
  arXiv:1603.05027 (2016)

  Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy,
  Alan L. Yuille
  DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
  Atrous Convolution, and Fully Connected CRFs
  arXiv:1606.00915 (2016)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

default_config = {
    'image_size': 224,
    'num_classes': 1000,
    }

class VggModel(nn.Module):
  def __init__(self, 
               features, 
               num_classes=default_config['num_classes'],
               init_weights=True):
    super(VggModel, self).__init__()
    self.features = features
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
            )

    if init_weights:
      self._initialize_weights()

  def forward(self, inputs):
    output = self.features(inputs)
    output = self.avgpool(output)
    output = output.view(output.size(0), -1)
    output = self.classifier(output)

    return output
  
  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
  layers = []
  depth = 3
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(depth, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      depth = v
  return nn.Sequential(*layers)

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 
        512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 
        'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 
        512, 512, 'M', 512, 512, 512, 512, 'M'],

def create_vgg11(**kwargs):
  model = VggModel(make_layers(cfg['vgg11']), **kwargs)
  return model

def create_vgg11_bn(**kwargs):
  model = VggModel(make_layers(cfg['vgg11'], batch_norm=True), **kwargs)
  return model

def create_vgg13(**kwargs):
  model = VggModel(make_layers(cfg['vgg13']), **kwargs)
  return model

def create_vgg13_bn(**kwargs):
  model = VggModel(make_layers(cfg['vgg13'], batch_norm=True), **kwargs)
  return model

def create_vgg16(**kwargs):
  model = VggModel(make_layers(cfg['vgg16']), **kwargs)
  return model

def create_vgg16_bn(**kwargs):
  model = VggModel(make_layers(cfg['vgg16'], batch_norm=True), **kwargs)
  return model

def create_vgg19(**kwargs):
  model = VggModel(make_layers(cfg['vgg19']), **kwargs)
  return model

def create_vgg19_bn(**kwargs):
  model = VggModel(make_layers(cfg['vgg19'], batch_norm=True), **kwargs)
  return model
