from cifar10 import *
from argparse import Namespace

def fine_tune_all(log_file, postfix, adapt=False):
    args = get_args()  # default args
    args.log = True
    args.log_file = log_file
    args.exp_name = f"fine_tune_all_{postfix}"

    args.batch_size = 64
    args.n_accumulation_steps = 8
    args.sigma = 1.
    args.odometer = True
    args.enable_bn = True
    args.lr = 0.01
    args.epochs = 50
    args.wd = 0
    args.model_type = 'ResNet18'
    args.fine_tune = 'all'
    args.gamma_mult = 390
    args.log_dir = f"ResNet18_sgd_lr.01_wd0_e50_b512_s1_odometer_finetune-all_gammamult390_{postfix}"

    if adapt:
        args.sigma = 2.
        args.adaptive = 'noise'
        args.adaptive_metric = 'train_acc'
        args.adaptive_strategy = 'down'
        args.adaptation_additive = .1
        args.maximum_factor = 2.
        args.adaptation_period = 1
        args.move_down_period = -1
        args.log_dir = f"ResNet18_sgd_lr.01_wd0_e100_b512_s2_odometer_adaptive-noise-down-.1-1-2_period1-trainacc_finetune-all_{postfix}"

    main(args)

def fine_tune_last(log_file, postfix, adapt=False):
    args = get_args()  # default args
    args.log = True
    args.log_file = log_file
    args.exp_name = f"fine_tune_last_{postfix}"

    args.batch_size = 64
    args.n_accumulation_steps = 8
    args.sigma = 1.
    args.odometer = True
    args.enable_bn = True
    args.lr = 0.01
    args.epochs = 50
    args.wd = 0
    args.model_type = 'ResNet18'
    args.fine_tune = 'last'
    args.gamma_mult = 1
    args.log_dir = f"ResNet18_sgd_lr.01_wd0_e50_b512_s1_odometer_finetune-last_gammamult1_{postfix}"

    if adapt:
        args.sigma = 2.
        args.adaptive = 'noise'
        args.adaptive_metric = 'train_acc'
        args.adaptive_strategy = 'down'
        args.adaptation_additive = .1
        args.maximum_factor = 2.
        args.adaptation_period = 1
        args.move_down_period = -1
        args.log_dir = f"ResNet18_sgd_lr.01_wd0_e100_b512_s2_odometer_adaptive-noise-down-.1-1-2_period1-trainacc_finetune-last_{postfix}"

    main(args)

if __name__ == "__main__":
    repeat = 5
    log_file = "exp_results/fine_tune_exp.json"
    for r in range(repeat):
        fine_tune_all(log_file, postfix=r)
        fine_tune_last(log_file, postfix=r)
    log_file = "exp_results/fine_tune_adapt_exp.json"
    for r in range(repeat):
        fine_tune_all(log_file, postfix=r, adapt=True)
        fine_tune_last(log_file, postfix=r, adapt=True)

