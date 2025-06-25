# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.utils.clip_grad import dispatch_clip_grad
import utils
import torch, numpy

#from apex import amp, optimizers, parallel
def train_one_epoch(model: torch.nn.Module, criterion: None,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    fp32=False, mini_batch=0):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if mini_batch > 0:
            samples = torch.split(samples, mini_batch, dim=0)
            targets = torch.split(targets, mini_batch, dim=0)
            loss = []
            with torch.cuda.amp.autocast(enabled=not fp32):
                for s, t in zip(samples, targets):
                    o = model(s)
                    l = criterion(o, t)
                    loss.append(l)
                    if loss_scaler is not None:
                        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                        loss_scaler._scaler.scale(l).backward(create_graph=is_second_order)
                        if max_norm is not None:
                            assert model.parameters() is not None
                            loss_scaler._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                            dispatch_clip_grad(model.parameters(), max_norm)
                loss = torch.mean(torch.stack(loss))
        # with torch.cuda.amp.autocast():
        #     outputs = model(samples)
        #     loss = criterion(samples, outputs, targets)
        else:
            with torch.cuda.amp.autocast(enabled=not fp32):
                outputs = model(samples)
                loss = criterion(outputs, targets)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            # print("Loss is {}, stopping training".format(loss_value))
            # sys.exit(1)
            print("Loss is {}, skip this step".format(loss_value))
            samples = samples.cpu().numpy()
            if True in numpy.isnan(samples):
                print("NaN in samples")
            elif True in numpy.isinf(samples):
                print("Inf in samples")
            targets = targets.cpu().numpy()
            if True in numpy.isnan(targets):
                print("NaN in targets")
            elif True in numpy.isinf(targets):
                print("Inf in targets")
            outputs = outputs.detach().cpu().numpy()
            if True in numpy.isnan(outputs):
                print("NaN in outputs")
            elif True in numpy.isinf(outputs):
                print("Inf in outputs")
            model_para = model.state_dict()
            for key in model_para.keys():
                if True in numpy.isnan(model_para[key].cpu().numpy()):
                    print("NaN in {}".format(key))
                elif True in numpy.isinf(model_para[key].cpu().numpy()):
                    print("Inf in {}".format(key))
            sys.exit(255)

            # sys.exit(1)
            # optimizer.zero_grad()
            # del loss, outputs, targets
            # metric_logger.update(loss=0.)
            # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # continue
        

        # this attribute is added by timm on one optimizer (adahessian)
        if mini_batch > 0:
            if loss_scaler is not None:
                loss_scaler._scaler.step(optimizer)
                loss_scaler._scaler.update()
        else:
            if loss_scaler is not None:
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                            parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, dataset, mini_batch):
    
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        batch_size = images.shape[0]

        # compute output
        with torch.cuda.amp.autocast():
            if mini_batch > 0:
                images = torch.split(images, mini_batch, dim=0)
                output = []
                for im in images:
                    o = model(im)
                    output.append(o)
                output = torch.cat(output, dim=0)
                loss = criterion(output, target)
            else:
                output = model(images)
                loss = criterion(output, target)
        
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(float(acc1), n=batch_size)
        metric_logger.meters['acc5'].update(float(acc5), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
