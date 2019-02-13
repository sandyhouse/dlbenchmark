# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Model configurations for CNN benchmarks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.cnn import alexnet_model
from models.cnn import googlenet_model
from models.cnn import resnet_model
from models.cnn import trivial_model
from models.cnn import vgg_model

_model_name_to_imagenet_model = {
    'vgg11': vgg_model.Vgg11Model,
    'vgg16': vgg_model.Vgg16Model,
    'vgg19': vgg_model.Vgg19Model,
    'googlenet': googlenet_model.GooglenetModel,
    'alexnet': alexnet_model.AlexnetModel,
    'trivial': trivial_model.TrivialModel,
    'resnet50': resnet_model.create_resnet50_model,
    'resnet50_v1.5': resnet_model.create_resnet50_v1_5_model,
    'resnet50_v2': resnet_model.create_resnet50_v2_model,
    'resnet101': resnet_model.create_resnet101_model,
    'resnet101_v2': resnet_model.create_resnet101_v2_model,
    'resnet152': resnet_model.create_resnet152_model,
    'resnet152_v2': resnet_model.create_resnet152_v2_model,
}


_model_name_to_cifar_model = {
    'alexnet': alexnet_model.AlexnetCifar10Model,
    'resnet20': resnet_model.create_resnet20_cifar_model,
    'resnet20_v2': resnet_model.create_resnet20_v2_cifar_model,
    'resnet32': resnet_model.create_resnet32_cifar_model,
    'resnet32_v2': resnet_model.create_resnet32_v2_cifar_model,
    'resnet44': resnet_model.create_resnet44_cifar_model,
    'resnet44_v2': resnet_model.create_resnet44_v2_cifar_model,
    'resnet56': resnet_model.create_resnet56_cifar_model,
    'resnet56_v2': resnet_model.create_resnet56_v2_cifar_model,
    'resnet110': resnet_model.create_resnet110_cifar_model,
    'resnet110_v2': resnet_model.create_resnet110_v2_cifar_model,
    'trivial': trivial_model.TrivialCifar10Model,
}


def _get_model_map(dataset_name):
  """Get name to model map for specified dataset."""
  if dataset_name == 'cifar10':
    return _model_name_to_cifar_model
  elif dataset_name in ('imagenet', 'synthetic'):
    return _model_name_to_imagenet_model
  else:
    raise ValueError('Invalid dataset name: %s' % dataset_name)


def get_model_config(model_name, dataset, params):
  """Map model name to model network configuration."""
  model_map = _get_model_map(dataset.name)
  if model_name not in model_map:
    raise ValueError('Invalid model name \'%s\' for dataset \'%s\'' %
                     (model_name, dataset.name))
  else:
    return model_map[model_name](params=params)


def register_model(model_name, dataset_name, model_func):
  """Register a new model that can be obtained with `get_model_config`."""
  model_map = _get_model_map(dataset_name)
  if model_name in model_map:
    raise ValueError('Model "%s" is already registered for dataset "%s"' %
                     (model_name, dataset_name))
  model_map[model_name] = model_func
