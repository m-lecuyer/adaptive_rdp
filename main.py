#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs CIFAR10 training with differential privacy.
"""

import argparse
import os
import shutil
import math
import json

from models.resnet9 import ResNet9

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.tensorboard as tensorboard
import torchcsprng as prng
import torchvision.models as models
import torchvision.transforms as transforms
#  from opacus import PrivacyEngine
from adaptive_privacy_engine import AdaptivePrivacyEngine, PrivacyFilterEngine, PrivacyOdometerEngine
from opacus.utils import stats
from opacus.utils.module_modification import convert_batchnorm_modules
from torchvision.datasets import CIFAR10
from tqdm import tqdm

def accuracy(preds, labels):
    return (preds == labels).mean()

def set_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_to_no_grad(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        #  m.apply(set_no_grad)
        for param in m.parameters():
            param.requires_grad = False

def train(args, model, train_loader, optimizer, epoch, device, privacy_engine):
    model.train()

    if args.fine_tune in ['last', 'all'] and args.enable_bn and not args.disable_dp:
        model.apply(set_bn_to_eval)
    if args.fine_tune == 'last':
        model.apply(set_bn_to_eval)

    criterion = nn.CrossEntropyLoss(reduction=args.loss_reduction)

    losses = []
    top1_acc = []

    iter_n = len(train_loader)
    pbar = tqdm(train_loader)
    for i, (images, target) in enumerate(pbar):
        if args.filter is not None and privacy_engine.halt():
            return

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc1)

        stats.update(
            stats.StatType.TRAIN,
            'accuracy',
            acc1=acc1,
        )

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        if ((i + 1) % privacy_engine.n_accumulation_steps == 0) or ((i + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer.virtual_step()

        if i % args.print_freq == 0:
            if not args.disable_dp:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent()
                stats.update(stats.StatType.PRIVACY, epsilon=epsilon, delta=args.delta, alpha=best_alpha)
                pbar.set_description(
                    f"Epoch: {epoch} "
                    f"Loss: {np.mean(losses):.2f} "
                    f"Acc@1: {np.mean(top1_acc[-5:]):.3f} "
                    f"(ε={epsilon:.2f}, δ={args.delta}), α={best_alpha}"
                )
            else:
                pbar.set_description(
                    f"Epoch: {epoch} "
                    f"Loss: {np.mean(losses):.2f} "
                    f"Acc@1: {np.mean(top1_acc[-5:]):.3f} "
                )


def test(args, model, test_loader, device, stat_name="Acc@1"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest: " f"Loss: {np.mean(losses):.6f} " f"{stat_name}: {top1_avg :.6f}")
    return np.mean(top1_acc)

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "-na",
        "--n_accumulation_steps",
        default=1,
        type=int,
        metavar="N",
        help="number of mini-batches to accumulate into an effective batch",
    )
    parser.add_argument(
        "--loss-reduction",
        default="mean",
        type=str,
        help="Loss reduction (mean, sum, none)",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=5e-4,
        type=float,
        metavar="W",
        help="SGD weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        metavar="GAMMA",
        help="Learning rate decay factor (e.g. for MultiStep)",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--enable-bn",
        action="store_true",
        default=False,
        help="Enables batch norm (for non DP model only)",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-6,
        metavar="D",
        help="Target delta (default: 1e-6)",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where CIFAR10 is/will be stored",
    )
    parser.add_argument(
        "--log-dir", type=str, default="", help="Where Tensorboard log will be stored"
    )

    parser.add_argument(
        "--filter",
        type=float,
        default=None,
        metavar="F",
        help="Use a PrivacyFilter with the argument upper bound (default None)",
    )
    parser.add_argument(
        "--odometer",
        default=False,
        action="store_true",
        help="Use a PrivacyOdometer (default False)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ResNet9",
        help="Model to use (ResNet9, ResNet18, PreActResNet18)",
    )
    parser.add_argument(
        "--fine-tune",
        type=str,
        default="No",
        help="Fine tuning approach (all, last) or anything else to train from scratch",
    )

    parser.add_argument(
        "--adaptive",
        type=str,
        default="None",
        help="Adaptive query budget, with value in (batch, noise)",
    )
    parser.add_argument(
        "--adaptive-metric",
        type=str,
        default="train_acc",
        help="Metric to decide how to adapt, in (train_acc, test_acc, grads_bellow_clip)",
    )
    parser.add_argument(
        "--adaptive-metric-std",
        type=float,
        default=100.,
        help="Noise factor to use when compution the adaptation metric (default 100.)",
    )
    parser.add_argument(
        "--adaptation-period",
        type=int,
        default=10,
        help="Number of epochs between adaptation decisions (default 10)",
    )
    parser.add_argument(
        "--move-down-period",
        type=int,
        default=50,
        help="Numer of epochs when adaptation can only go down (default 50). Set to -1 to disable",
    )
    parser.add_argument(
        "--adaptive-strategy",
        type=str,
        default="None",
        help="Adaptation strategy in (down, updown)",
    )
    parser.add_argument(
        "--maximum-factor",
        type=float,
        default=2.,
        help="The maximum factor from starting value (default 2.)",
    )
    parser.add_argument(
        "--minimum-factor",
        type=float,
        default=.5,
        help="The minumum factor from starting value (default .5)",
    )
    parser.add_argument(
        "--adaptation-additive",
        type=float,
        default=2.,
        help="Increase/decrease factor for adaptation (default 2.)",
    )

    parser.add_argument(
        "--log",
        action="store_true",
        default=False,
        help="Log data in log-file",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="exp",
        help="log-file name",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="exp",
        help="exp name in the log",
    )

    args = parser.parse_args()
    return args

def main(args):
    if args.disable_dp and args.n_accumulation_steps > 1:
        raise ValueError("Virtual steps only works with enabled DP")

    if args.enable_bn and not (args.disable_dp or args.fine_tune in ['last', 'all']):
        raise ValueError("Batch norm only works without DP or with --fine-tune last")

    if args.fine_tune in ['all', 'last']:
        fine_tune = True
    else:
        fine_tune = False

    # The following few lines, enable stats gathering about the run
    # 1. where the stats should be logged
    writer = tensorboard.SummaryWriter(os.path.join("/tmp/stat", args.log_dir))
    stats.set_global_summary_writer(writer)
    # 2. enable stats
    stats.add(
        # stats on privacy budget spent
        stats.Stat(stats.StatType.PRIVACY, "privacy"),
        # stats on training accuracy
        stats.Stat(stats.StatType.TRAIN, "accuracy", frequency=0.1),
        stats.Stat(stats.StatType.TRAIN, "train_accuracy", frequency=1.),
        stats.Stat(stats.StatType.TRAIN, "metadata", frequency=1.),
        # stats on validation accuracy
        stats.Stat(stats.StatType.TEST, "accuracy", frequency=1.),
    )

    # The following lines enable stat gathering for the clipping process
    # and set a default of per layer clipping for the Privacy Engine
    clipping = {"clip_per_layer": False, "enable_stat": True}

    generator = (
        prng.create_random_device_generator("/dev/urandom") if args.secure_rng else None
    )
    if fine_tune:
        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalize = [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        augmentations = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        normalize = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]

    train_transform = transforms.Compose(
        #  augmentations + normalize if args.disable_dp else normalize
        augmentations + normalize
        #  normalize
    )

    test_transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        generator=generator,
        pin_memory=True,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2*args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )


    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2*args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    device = torch.device(args.device)

    if args.fine_tune in ['all', 'last']:
        if args.model_type == 'ResNet18':
            model = models.resnet18(pretrained=True)
        else:
            raise ValueError("Model not recognized for fine tuning. Please check spelling")

        if args.fine_tune == 'last':
            #  model.apply(set_no_grad)
            for param in model.parameters():
                param.requires_grad = False
        if args.fine_tune == 'all' and not args.disable_dp and args.enable_bn:
            model.apply(set_bn_to_no_grad)

        model.fc = nn.Linear(model.fc.in_features, 10)
    else:
        if args.model_type == 'ResNet9':
            model = ResNet9()
        elif args.model_type == 'ResNet18':
            model = ResNet18
        elif args.model_type == 'PreActResNet18':
            model = PreActResNet18
        else:
            raise ValueError("Model not recognized. Please check spelling")

    if not args.enable_bn:
        if fine_tune and args.fine_tune != 'all':
            raise ValueError("When fine tuning without all, cannot disable batch norm")
        model = convert_batchnorm_modules(model)

    print(model)

    images, labels = next(iter(train_loader))
    writer.add_graph(model, images)

    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if not args.disable_dp:
        if args.filter is not None:
            privacy_engine = PrivacyFilterEngine(
                args.filter,
                args.delta,
                model,
                batch_size=args.batch_size * args.n_accumulation_steps,
                n_accumulation_steps=args.n_accumulation_steps,
                sample_size=len(train_dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                secure_rng=args.secure_rng,
                **clipping,
            )
        elif args.odometer:
            privacy_engine = PrivacyOdometerEngine(
                args.delta,
                model,
                batch_size=args.batch_size * args.n_accumulation_steps,
                n_accumulation_steps=args.n_accumulation_steps,
                sample_size=len(train_dataset),
                alphas=[2 + x / 4 for x in range(1, 33)] + [16, 32],
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                secure_rng=args.secure_rng,
                **clipping,
            )
        else:
            privacy_engine = AdaptivePrivacyEngine(
                model,
                batch_size=args.batch_size * args.n_accumulation_steps,
                n_accumulation_steps=args.n_accumulation_steps,
                sample_size=len(train_dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
                secure_rng=args.secure_rng,
                **clipping,
            )
        privacy_engine.attach(optimizer)
    else:
        privacy_engine = None

    n_grads_bellow_clip_std = 100.
    max_n_grads_bellow_clip = 0
    last_increase_in_max_n_grads_bellow_clip = 0
    epochs_without_increase_in_n_grads_bellow_clip = 30
    #
    #
    best_acc1 = 0
    best_top1_acc = 0
    best_top1_acc_epoch = 0
    #
    best_top1_acc_train = 0
    best_top1_acc_train_epoch = 0
    #
    adaptive_metric = 0.
    last_adaptive_metric_max = 0
    #
    n_accumulation_steps = args.n_accumulation_steps
    sigma = args.sigma
    #
    for epoch in range(args.start_epoch, args.epochs):
        stats.update(
            stats.StatType.TRAIN, 'metadata',
            batch_size=args.batch_size * n_accumulation_steps,
            noise_multiplier=sigma,
            adaptive_metric=adaptive_metric,
        )

        train(args, model, train_loader, optimizer, epoch, device, privacy_engine)

        train_acc = 0
        std = args.adaptive_metric_std
        if args.adaptive_metric == 'train_acc':
            adaptive_metric = test(args, model, train_eval_loader, device, stat_name='TrainAcc@1') * len(train_dataset)
            adaptive_metric = torch.normal(torch.tensor(adaptive_metric), std).detach()
            train_acc = adaptive_metric
        else:
            raise ValueError("Supported adaptive metric: train_acc")

        adapt = False
        if args.adaptive in ['batch', 'noise']:
            adapt = (epoch+1) % args.adaptation_period == 0

        if adapt:
            privacy_engine.add_query_to_ledger(1., std, 1)

            significant_increase = last_adaptive_metric_max < adaptive_metric - 3*std
            significant_decrease = last_adaptive_metric_max - 3*std > adaptive_metric

            if significant_increase:
                last_adaptive_metric_max = adaptive_metric

            steps_per_epoch = math.ceil(len(train_loader) / n_accumulation_steps)
            can_go_up = args.move_down_period <= 0 or not privacy_engine.halt(steps=steps_per_epoch*args.move_down_period)
            additive = 0.
            if args.adaptive_strategy == 'down':
                if not significant_increase:
                    # go down (less private) linearly
                    additive = args.adaptation_additive
            elif args.adaptive_strategy == 'updown':
                if significant_increase:
                    # go up (more private) linearly
                    if can_go_up:
                        additive = -args.adaptation_additive
                else:
                    # go down (less private) linearly
                    additive = args.adaptation_additive

            if args.adaptive == 'batch' and additive != 0:
                n_accumulation_steps += additive
                n_accumulation_steps = max(1, int(n_accumulation_steps))
                n_accumulation_steps = min(n_accumulation_steps, int(args.maximum_factor * args.n_accumulation_steps))
                n_accumulation_steps = max(n_accumulation_steps, int(args.minimum_factor * args.n_accumulation_steps))

                privacy_engine.update_batch_size(args.batch_size * n_accumulation_steps, n_accumulation_steps)
            if args.adaptive == 'noise' and additive != 0:
                sigma -= additive
                sigma = max(args.sigma / args.maximum_factor, sigma)
                sigma = min(args.sigma / args.minimum_factor, sigma)
                privacy_engine.update_noise_multiplier(sigma)

        top1_acc = test(args, model, test_loader, device)
        is_best_top1_acc = top1_acc > best_top1_acc
        if is_best_top1_acc:
            best_top1_acc = top1_acc
            best_top1_acc_epoch = epoch

        stats.update(
            stats.StatType.TEST,
            acc1=top1_acc,
            acc1_epochs_since_best=epoch - best_top1_acc_epoch,
        )

        # remember best acc@1 and save checkpoint
        is_best = top1_acc > best_acc1
        best_acc1 = max(top1_acc, best_acc1)

        if args.log:
            if os.path.exists(args.log_file):
                with open(args.log_file, 'r') as f:
                    log = json.loads(f.read())
            else:
                with open(args.log_file, 'w+') as f: pass
                log = {}

            if args.exp_name not in log:
                log[args.exp_name] = {
                    'epoch': [],
                    'train_acc': [],
                    'test_acc': [],
                    'steps_in_epoch': [],
                    'n_accumulation_steps': [],
                    'batch_size': [],
                    'sigma': [],
                }

            log[args.exp_name]['epoch'].append(epoch)
            log[args.exp_name]['train_acc'].append(train_acc.item())
            log[args.exp_name]['test_acc'].append(top1_acc.item())
            log[args.exp_name]['steps_in_epoch'].append(len(train_loader) // n_accumulation_steps)
            log[args.exp_name]['n_accumulation_steps'].append(n_accumulation_steps)
            log[args.exp_name]['batch_size'].append(args.batch_size * n_accumulation_steps)
            log[args.exp_name]['sigma'].append(sigma)

            with open(args.log_file, 'w') as f:
                f.write(json.dumps(log))

        if args.filter is not None and privacy_engine.halt():
            break

    writer.flush()
    stats.clear()
    writer.close()


if __name__ == "__main__":
    args = get_args()
    main(args)
