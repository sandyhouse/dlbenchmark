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
import numpy as np

import pytorch.models.cnn.alexnet_model as alexnet
import pytorch.models.cnn.resnet_model as resnet
#import pytorch.models.cnn.googlenet_model as googlenet
import pytorch.models.cnn.vgg_model as vgg

import torch
import torch.multiprocessing as mp
import torchvision

def get_optimizer(model, params, learning_rate):
  """Returns the optimizer that should be used based on params."""
  if params.optimizer == 'momentum':
    opt = torch.optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=params.momentum, nesterov=True)
  elif params.optimizer == 'sgd':
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

MODEL_CREATOR = {
        'alexnet': alexnet.AlexnetModel,
        'resnet50': resnet.create_resnet50,
        'vgg16': vgg.create_vgg16,
    }
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

    self.model = self.params.model
    self.batch_size_per_device = self.params.batch_size
    self.batch_size = self.params.batch_size

    self.num_gpus = self.params.num_gpus
    if self.num_gpus:
      self.batch_size = self.batch_size_per_device * self.params.num_gpus

    self.do_train = self.params.do_train
    self.do_eval = self.params.do_eval

    self.num_epochs = self.params.num_epochs

    self.use_synthetic_data = False if self.params.data_dir else True
    self.data_dir = self.params.data_dir

    self.use_fp16 = self.params.use_fp16

    if self.use_fp16:
      self.data_type = torch.float16
    else:
      self.data_type = torch.float32

    self.data_format = self.params.data_format
    self.optimizer = self.params.optimizer
    self.all_reduce_spec = self.params.all_reduce_spec

    self.params.distributed = True if self.params.ip_list else False
    self.params.world_size = 1
    if self.params.ip_list:
      ips = self.params.ip_list.split(',')
      self.params.world_size = len(ips)
      self.params.rank = self.params.job_index
      address = ips[0] + ":63271"
      # url used to set up distributed environment
      self.params.dist_url = "tcp://" + address
      self.params.dist_backend = "nccl"
    self.params.multiprocessing_distributed = False
    if self.num_gpus > 1:
      self.params.multiprocessing_distributed = True
    self.params.workers = 4 # Number of data loading workers.

    self.print_info()
    
  def print_info(self):
    """Print basic information."""
    dataset_name = "ImageNet-synthetic" if self.use_synthetic_data else (
                   "ImageNet")
    mode = ""
    if self.do_train:
      mode += 'train '
    if self.do_eval:
      mode += 'eval '

    print()
    print('Model:       %s' % self.model)
    print('Dataset:     %s' % dataset_name)
    print('Mode:        %s' % mode)
    print('Batch size:  %s global (per machine)' % (
           self.batch_size))
    print('             %s per device' % (self.batch_size_per_device))
    print('Num GPUs:    %d per worker' % (self.num_gpus))
    print('Num epochs:  %d' % self.num_epochs)
    print('Data format: %s' % self.data_format)
    print('Optimizer:   %s' % self.optimizer)
    print('AllReduce:   %s' % self.all_reduce_spec)
    print('=' * 30)

  def run(self):
    """Run the benchmark task assigned to this process."""
    if self.params.multiprocessing_distributed:
      self.params.world_size = self.num_gpus * self.params.world_size
      mp.spawn(main_worker, nprocs=self.num_gpus, 
               args=(self.model, self.num_gpus, self.distributed, self.params))
    else:
      main_worker(self.model, self.num_gpus, self.params.distributed,
              self.params)

def main_worker(model, num_gpus, distributed, params):
    model = MODEL_CREATOR[model]()
    if num_gpus == 1 and not distributed:
      model = model.cuda(0)
    else: #self.distributed:
      model.cuda()
      model = torch.nn.parallel.DistributedDataParallel(model)
      self.params.rank = self.params.rank * num_gpus + gpu
      dist.init_process_group(backend=params.dist_backend,
              init_method=params.dist_url,
              world_size=params.world_size,
              rank=params.rank)
#    else:
#      if self.params.model.startswith('alexnet') or self.params.model.startswith('vgg'):
#        model.features = torch.nn.DataParallel(model.features)
#        model.cuda()
#      else:
#        model = torch.nn.DataParallel(model).cuda()
#    
    criterion = torch.nn.CrossEntropyLoss().cuda(0)
    optimizer = get_optimizer(params.model, params, params.init_learning_rate)
#
    if params.data_dir:
      checkpoint_file = os.path.join(params.data_dir, 'checkpoint.ckt')
      if os.path.isfile(checkpoint_file):
        print("Loading checkpoint: {}".format(checkpoint))
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        top_1_acc = checkpoint['top_1_acc1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint {} at epoch {}"
                .format(checkpoint, start_epoch))

    train_data_dir = os.path.join(params.data_dir, 'train')
    if params.do_eval:
      eval_data_dir = os.path.join(self.params.data_dir, 'eval')

    normalized = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])

    train_data = torchvision.datasets.ImageFolder(
            train_data_dir,
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalized]))

    if distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(
              train_data)
    else:
      train_sampler = None

    train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.params.batch_size, 
            shuffle=(train_sampler is None),
            num_workers=params.workers, pin_memory=True, 
            sampler=train_sampler)

    if params.eval:
      eval_data = torchvision.datasets.ImageFolder(
            eval_data_dir,
            torchvision.transforms.Compose([
                torchvision.transforms.Resized(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                normalized]))
            
      eval_data_loader = torch.utils.data.DataLoader(
             eval_data, batch_size=self.params.batch_size, 
             shuffle=False,
             num_workers=params.workers, pin_memory=True)


    for epoch in range(start_epoch, params.num_epochs):
      epoch_start_time = time.time()
      if params.distributed:
        train_sampler.set_epoch(epoch)
      adjust_learning_rate(optimizer, epoch)

      # train for one epoch.
      train(train_loader, model, criterion, optimizer, epoch)

      print("Epoch {} ran time: {}".format(
          epoch, time.time() - epoch_start_time))
      acc1 = validate(val_loader, model, criterion)

      is_best = acc1 > top_1_acc
      top_1_acc = max(acc1, top_1_acc)
      save_checkpoint({
          'epoch': epoch + 1,
          'model': params.model,
          'state_dict': model.state_dict(),
          'best_acc1': top_1_acc1,
          'optimizer': optimizer.state_dict(),
          }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
  """Train for one epoch."""
  batch_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  model.train()

  end = time.time()
  for i, (inputs, label) in enumerate(train_loader):
    # measure data loading time
    data_time.update(time.time() - end)

    if self.params.num_gpus == 1:
      inputs = inputs.cuda(0, non_blocking=True)
    label = label.cuda(0, non_blocking=True)
    output = model(inputs)
    loss = criterion(output, label)

    acc1, acc5 = accuracy(output, label, topk=(1, 5))
    losses.update(loss.item(), inputs.size(0))
    top1.update(acc1[0], inputs.size(0))
    top5.update(acc5[0], inputs.size(0))

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    batch_time.update(time.time() - end)
    end = time.time()

    print("Epoch: [{0}][{1}/{2}]\t"
          "Time: {batch_time.val: .3f} ({batch_time.avg: .3f})\t"
          "Data: {data_time.val: .3f} ({data_time.avg: .3f})\t"
          "Loss: {loss.val: .4f} ({loss.avg: .4f})\t"
          "Acc@1: {top1.val: .3f} ({top1.avg: .3f})\t"
          "Acc@5: {top5.val: .3f} ({top5.avg: .3f})\t".format(
              epoch, i, len(train_loader), batch_time=batch_time,
              data_time=data_time, loss=losses, top1=top1, top5=top5))

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
      losses = criterion(output, label)

      acc1, acc5 = accuracy(output, label, topk=(1, 5))
      losses.update(losses.item(), inputs.size(0))
      top1.update(acc1[0], inputs.size(0))
      top1.update(acc5[0], inputs.size(0))

      batch_time.update(time.time() - end)
      end = time.end()
      print("Test: [{0}/{1}]\t"
            "Time: {batch_time.val: .3f} ({batch_time.avg: .3f})\t"
            "Loss: {loss.val: .4f} ({loss.avg: .4f})\t"
            "Acc@1: {top1.val: .3f} ({top1.avg: .3f})\t"
            "Acc@5: {top5.val: .3f} ({top5.avg: .3f})\t".format(
                i, len(train_loader), batch_time=batch_time,
                loss=losses, top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg: .3f} Acc@5 {top5.avg: .3f}'
            .format(top1=top1, top5=top5))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
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
