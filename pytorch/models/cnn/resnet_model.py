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

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, channels_in, depth, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels_in, depth, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(depth)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=1, padding=1,
                           bias=False)
    self.bn2 = nn.BatchNorm2d(depth)
    self.downsample = downsample
    self.stride = stride

  def forward(self, inputs):
    shortcut = inputs

    out = self.conv1(inputs)
    out = bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    
    if self.downsample is not None:
      shortcut = self.downsample(inputs)

    out += shortcut
    out = self.relu(out)

    return out

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, channels_in, depth, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(channels_in, depth, kernel_size=1, stride=1,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(depth)
    self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=stride, 
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(depth)
    self.conv3 = nn.Conv2d(depth, depth * self.expansion, kernel_size=1, 
                           stride=1, bias=False)
    self.bn3 = nn.BatchNorm2d(depth * self.expansion)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, inputs):
    shortcut = inputs

    out = self.conv1(inputs)
    out = bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)
    
    if self.downsample is not None:
      shortcut = self.downsample(inputs)

    out += shortcut
    out = self.relu(out)

    return out


class ResnetModel(nn.Module):
  def __init__(self, 
               block, 
               layers_count, 
               num_classes=default_config['num_classes'],
               zero_init_residual=False):
    super(ResnetModel, self).__init__()
    self.channels_in = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.layer1 = self._make_block(block, 64, layers_count[0])
    self.layer2 = self._make_block(block, 128, layers_count[1], stride=2)
    self.layer3 = self._make_block(block, 256, layers_count[2], stride=2)
    self.layer4 = self._make_block(block, 512, layers_count[3], stride=2)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nolinearity='relu')
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    if zero_init_residual:
      for m in self.modules():
        if isinstance(m, Bottleneck):
          nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock):
          nn.init.constant_(m.bn2.weight, 0)
  
  def _make_block(self, block, depth, num_blocks, stride=1):
    downsample = None
    if stride != 1 or self.channels_in != depth * block.expansion:
      downsample = nn.Sequential(
              nn.Conv2d(self.channels_in, depth * block.expansion, 
                  kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm2d(depth * block.expansion),
              )

    layers = []
    layers.append(block(self.channels_in, depth, stride, downsample))
    self.channels_in = depth * block.expansion
    for _ in range(1, num_blocks):
      layers.append(block(self.channels_in, depth))

    return nn.Sequential(*layers)

  def forward(self, inputs):
    output = self.conv1(inputs)
    output = self.bn1(output)
    output = self.relu(output)
    output = self.pool(output)

    output = self.layer1(output)
    output = self.layer2(output)
    output = self.layer3(output)
    output = self.layer4(output)

    output = self.avgpool(output)
    output = output.view(x.size(0), -1)
    output = self.fc(output)

    return output

def create_resnet18(**kwargs):
  model = ResnetModel(BasicBlock, [2, 2, 2, 2], **kwargs)
  return model
    
def create_resnet34(**kwargs):
  model = ResnetModel(BasicBlock, [3, 4, 6, 3], **kwargs)
  return model

def create_resnet50(**kwargs):
  model = ResnetModel(Bottleneck, [3, 4, 6, 3], **kwargs)
  return model

def create_resnet101(**kwargs):
  model = ResnetModel(Bottleneck, [3, 4, 23, 3], **kwargs)
  return model

def create_resnet152(**kwargs):
  model = ResnetModel(Bottleneck, [3, 8, 36, 3], **kwargs)
  return model
