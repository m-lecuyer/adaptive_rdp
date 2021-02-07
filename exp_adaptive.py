from cifar10 import *
from argparse import Namespace
import opacus
import torch

def baseline(log_file, postfix):
    args = get_args()  # default args
    args.log = True
    args.log_file = log_file
    args.exp_name = f"baseline_{postfix}"

    args.batch_size = 128
    args.n_accumulation_steps = 4
    args.sigma = 1.
    args.lr = 0.05
    args.epochs = 100
    args.weight_decay = 0
    args.model_type = 'ResNet9'
    args.log_dir = f"ResNet9_sgd_lr.05_wd0_e100_b512_s1_{postfix}"

    main(args)

    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    rdp = opacus.privacy_analysis.compute_rdp(q=args.batch_size*args.n_accumulation_steps/50000, noise_multiplier=args.sigma, steps=98*args.epochs, orders=alphas)
    eps = opacus.privacy_analysis.get_privacy_spent(alphas, rdp, args.delta)[0]
    return eps

def adapt_noise(log_file, postfix, epsilon, period_length):
    args = get_args()  # default args
    args.log = True
    args.log_file = log_file
    args.exp_name = f"adapt_noise_{postfix}"

    args.batch_size = 128
    args.n_accumulation_steps = 4
    args.sigma = 1.
    args.lr = 0.05
    args.epochs = 500
    args.weight_decay = 0
    args.model_type = 'ResNet9'
    args.filter = eps
    args.adaptive = 'noise'
    args.adaptive_metric = 'train_acc'
    args.adaptive_strategy = 'updown'
    args.adaptation_additive = .1
    args.maximum_factor = 1.
    args.adaptation_period = period_length
    args.move_down_period = 50
    args.log_dir = f"ResNet9_sgd_lr.05_wd0_e300_filter{epsilon}_adaptive-noise-tcp3-.1-1.25-1-period{period_length}-traincc_b512_{postfix}"

    main(args)

def adapt_batch(log_file, postfix, epsilon, period_length):
    args = get_args()  # default args
    args.log = True
    args.log_file = log_file
    args.exp_name = f"adapt_batch_{postfix}"

    args.batch_size = 128
    args.n_accumulation_steps = 4
    args.sigma = 1.
    args.lr = 0.05
    args.epochs = 500
    args.weight_decay = 0
    args.model_type = 'ResNet9'
    args.filter = eps
    args.adaptive = 'batch'
    args.adaptive_metric = 'train_acc'
    args.adaptive_strategy = 'updown'
    args.adaptation_additive = 1
    args.maximum_factor = 1.
    args.adaptation_period = period_length
    args.move_down_period = 50
    args.log_dir = f"ResNet9_sgd_lr.05_wd0_e300_filter{epsilon}_adaptive-batch-tcp3-1-2-1-period{period_length}-traincc_b512_{postfix}"

    main(args)

if __name__ == "__main__":
    repeat = 5
    log_file = "exp_results/adapt_exp.json"
    for r in range(repeat):
        eps = baseline(log_file, postfix=r)
        adapt_noise(log_file, postfix=r, period_length=10, epsilon=eps)
        adapt_batch(log_file, postfix=r, period_length=10, epsilon=eps)

