# --*-- coding:utf-8 --*--
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
"""PyTorch benchmarks for Convolutional Neural Networks (CNNs)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import shutil
import numpy as np

#impor pytorcumodels.cnn.alexnet_model as alexnet
#import pytorch.models.cnn.resnet_model as resnet
##import pytorch.models.cnn.googlenet_model as googlenet
#import pytorch.models.cnn.vgg_model as vgg

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_optimizer(model, params):
  """Returns the optimizer that should be used based on params."""
  learning_rate = params.init_learning_rate
  if params.optimizer == 'sgd':
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
  elif params.optimizer == 'adam':
    opt = torch.optim.Adam(model.parameters(),
                           lr=learning_rate, 
                           betas=(params.adam_beta1, params.adam_beta2), 
                           eps=params.adam_epsilon)
  else:
    raise ValueError('Optimizer "{}" was not recognized'.
                     format(params.optimizer))
  return opt

#MODEL_CREATOR = {
#        'alexnet': alexnet.AlexnetModel,
#        'resnet50': resnet.create_resnet50,
#        'vgg16': vgg.create_vgg16,
#    }

class BenchmarkCNN(object):
  """Class for benchmarking a cnn network."""

  def __init__(self, params):
    """Initialize BenchmarkCNN.

    Args:
      params: Params tuple, created by make_params_from_flags.
    Raises:
      ValueError: Unsupported params settings.
    """
    self.params = params

    #self.params.total_batch_size = self.params.batch_size
    

    #if self.params.num_gpus:
    #  self.params.batch_size *= self.params.num_gpus


    self.params.synthetic_data = False if self.params.data_dir else True

    if self.params.use_fp16:
      self.params.data_type = torch.float16
    else:
      self.params.data_type = torch.float32

    self.params.distributed = True if self.params.ip_list else False
    if self.params.ip_list:
      ips = self.params.ip_list.split(',')
      self.params.world_size = len(ips)
      self.params.rank = self.params.job_index
      address = ips[0] + ":63271"
      # url used to set up distributed environment
      self.params.dist_url = "tcp://" + address
      self.params.dist_backend = "nccl"
      dist.init_process_group(backend=self.params.dist_backend,
                              init_method=self.params.dist_url,
                              world_size=self.params.world_size,
                              rank=self.params.rank)
    self.params.workers = 20 # Number of data loading workers.

    self.print_info()
    
  def print_info(self):
    """Print basic information."""
    dataset_name = "ImageNet-synthetic" if self.params.synthetic_data else (
                   "ImageNet")
    mode = ""
    if self.params.do_train:
      mode += 'train '
    if self.params.do_eval:
      mode += 'eval '

    print()
    print('Model:       %s' % self.params.model)
    print('Dataset:     %s' % dataset_name)
    print('Mode:        %s' % mode)
    print('Batch size:  %s global (per machine)' % (
           self.params.batch_size))
    #print('             %s per device' % (self.params.batch_size))
    print('Num GPUs:    %d per worker' % (self.params.num_gpus))
    print('Num epochs:  %d' % self.params.num_epochs)
    print('Data format: %s' % self.params.data_format)
    print('Optimizer:   %s' % self.params.optimizer)
    print('=' * 30)

  def run(self):
    """Run the benchmark task assigned to this process."""
    print("Creating model '{}'".format(self.params.model))
    #model = MODEL_CREATOR[self.params.model]()
    model = models.__dict__[self.params.model]()
    if self.params.num_gpus == 1 and not self.params.distributed:
      torch.cuda.set_device(0)
      model = model.cuda(0)
    elif self.params.distributed:
      model.cuda()
      model = torch.nn.parallel.DistributedDataParallel(model)
    else:
      if self.params.model.startswith('alexnet') or (
                self.params.model.startswith('vgg')):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
      else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = get_optimizer(model, self.params)

    cudnn.benchmark = True
    train_dir = os.path.join(self.params.data_dir, 'train')
    eval_dir = os.path.join(self.params.data_dir, 'eval')

    normalized = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    train_dataset = torchvision.datasets.ImageFolder(
            train_dir,
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalized]))

    if self.params.distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
              train_dataset)
    else:
      train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.params.batch_size,
            shuffle=(train_sampler is None),
            num_workers=self.params.workers, pin_memory=True, 
            sampler=train_sampler)

    if self.params.do_eval:
      eval_dataset = torchvision.datasets.ImageFolder(
            eval_dir,
            torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalized]))
          
      eval_loader = torch.utils.data.DataLoader(
           eval_dataset, batch_size=self.params.batch_size, 
           shuffle=False,
           num_workers=self.params.workers, pin_memory=True)

    for epoch in range(int(self.params.num_epochs)):
      if self.params.distributed:
        train_sampler.set_epoch(epoch)
      adjust_learning_rate(optimizer, epoch, self.params)

      epoch_start_time = time.time()
      # train for one epoch.
      train(train_loader, model, criterion, optimizer, epoch, self.params)

      print("Epoch {} ran time: {}".format(
          epoch, time.time() - epoch_start_time))
      #if self.params.do_eval:
      #  acc1 = validate(eval_data_loader, model, criterion, self.params)

      #  is_best = acc1 > top_1_acc
      #  top_1_acc = max(acc1, top_1_acc)
      #  if self.params.model_dir:
      #    filename = os.path.join(self.params.model_dir, 'checkpoint.ckp')
      #    save_checkpoint({
      #      'epoch': epoch + 1,
      #      'model': self.params.model,
      #      'state_dict': model.state_dict(),
      #      'top_1_acc1': top_1_acc,
      #      'optimizer': optimizer.state_dict(),
      #      }, is_best, filename)

def train(train_loader, model, criterion, optimizer, epoch, params):
  """Train for one epoch."""
  batch_time = AverageMeter()
  data_time = AverageMeter()
  train_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  for i, (inputs, label) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if params.num_gpus == 1:
      inputs = inputs.cuda(0, non_blocking=True)
    label = label.cuda(0, non_blocking=True)
    output = model(inputs)
    loss = criterion(output, label)

    acc1, acc5 = accuracy(output, label, topk=(1, 5))
    losses.update(loss.item(), inputs.size(0))
    top1.update(acc1[0], inputs.size(0))
    top5.update(acc5[0], inputs.size(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    batch_time.update(time.time() - end)
    train_time.update(batch_time.val - data_time.val) 
    end = time.time()

    if i % 100 == 0:
      print("step: {}".format(i))
    #  print("=" * 30)
    #  print("Epoch: [{0}][{1}/{2}]\n"
    #        "Time: {batch_time.val: .3f} ({batch_time.avg: .3f})\n"
    #        "Data: {data_time.val: .3f} ({data_time.avg: .3f})\n"
    #        "Loss: {loss.val: .4f} ({loss.avg: .4f})\n"
    #        "Acc@1: {top1.val: .3f} ({top1.avg: .3f})\n"
    #        "Acc@5: {top5.val: .3f} ({top5.avg: .3f})\n".format(
    #            epoch, i, len(train_loader), batch_time=batch_time,
    #            data_time=data_time, loss=losses, top1=top1, top5=top5))
    #if i % 500 == 0:
    #  break
  average_examples_per_sec_without_data = params.batch_size / train_time.avg
  average_examples_per_sec = params.batch_size / batch_time.avg
  print("average_examples_per_sec_without_data: {}".format(
	average_examples_per_sec_without_data))
  print("average_examples_per_sec: {}".format(average_examples_per_sec))

def validate(val_loader, model, criterion, params):
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (inputs, label) in enumerate(val_loader):
      if params.num_gpus == 1:
        inputs = inputs.cuda(0, non_blocking=True)
      label = label.cuda(0, non_blocking=True)
      
      output = model(inputs)
      loss = criterion(output, label)

      acc1, acc5 = accuracy(output, label, topk=(1, 5))
      losses.update(loss.item(), inputs.size(0))
      top1.update(acc1[0], inputs.size(0))
      top1.update(acc5[0], inputs.size(0))

      batch_time.update(time.time() - end)
      end = time.time()
      #print("*" * 30)
      #print("Test: [{0}/{1}]\n"
      #      "Time: {batch_time.val: .3f} ({batch_time.avg: .3f})\n"
      #      "Loss: {loss.val: .4f} ({loss.avg: .4f})\n"
      #      "Acc@1: {top1.val: .3f} ({top1.avg: .3f})\n"
      #      "Acc@5: {top5.val: .3f} ({top5.avg: .3f})\n".format(
      #          i, len(val_loader), batch_time=batch_time,
      #          loss=losses, top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg: .3f} Acc@5 {top5.avg: .3f}'
            .format(top1=top1, top5=top5))
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.ckp'):
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, params):
  lr = params.init_learning_rate * (0.1 ** (epoch // 30))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

def accuracy(output, label, topk=(1,)):
  with torch.no_grad():
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0/batch_size))

    return res
