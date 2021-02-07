import opacus
import numpy as np
import types
import warnings
from typing import List, Optional, Tuple, Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, math

plt.style.use(['classic', 'seaborn-deep', 'seaborn-dark'])
sns.set(style='whitegrid')
cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linewidth=2
markersize=14
dataset_size = 50000

alphas=[2 + x / 4 for x in range(1, 33)] + [16, 32]

def get_rdp(sigma, real_batch_size, steps):
    if isinstance(sigma, list):
        rdp = opacus.privacy_analysis.compute_rdp(
            q=real_batch_size[0]/50000, noise_multiplier=sigma[0], steps=steps[0], orders=alphas
        )
        for i in range(1, len(sigma)):
            rdp += opacus.privacy_analysis.compute_rdp(
                q=real_batch_size[i]/50000, noise_multiplier=sigma[i], steps=steps[i], orders=alphas
            )
        return rdp
    else:
        return opacus.privacy_analysis.compute_rdp(
            q=real_batch_size/50000, noise_multiplier=sigma, steps=steps, orders=alphas
        )

def get_rdp_eps(sigma, real_batch_size, steps, rdp_init=None):
    rdp = get_rdp(sigma, real_batch_size, steps)
    if rdp_init is not None: rdp += rdp_init
    eps, alpha = opacus.privacy_analysis.get_privacy_spent(alphas, rdp, 1e-6)
    return eps, alpha, rdp

def get_odom_eps(sigma, real_batch_size, steps, gamma=None, rdp_init=None):
    if gamma is None:
        gamma = get_rdp(sigma, real_batch_size, 1)

    rdp = get_rdp(sigma, real_batch_size, steps)
    if rdp_init is not None: rdp += rdp_init

    _rdp = np.maximum(rdp, gamma)
    f = np.ceil(np.log2(_rdp / gamma))
    target_delta = 1e-6 / (2*np.power(f+1, 2)*len(alphas))
    _rdp = gamma * np.exp2(f)

    eps, alpha = _get_privacy_spent(_rdp, target_delta)

    idx = alphas.index(alpha)
    #  print(f"α={alpha} f={f[idx]}")

    return eps, alpha, rdp

def _get_privacy_spent(
    rdp: Union[List[float], float], delta: Union[List[float], float],
) -> Tuple[float, float]:
    orders_vec = np.atleast_1d(alphas)
    rdp_vec = np.atleast_1d(rdp)
    delta_vec = np.atleast_1d(delta)

    if len(orders_vec) != len(rdp_vec) or len(orders_vec) != len(delta_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
            f"\tdelta_vec = {delta_vec}\n"
        )

    eps = rdp_vec - np.log(delta) / (orders_vec - 1)

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    return eps[idx_opt], orders_vec[idx_opt]

def plot_odometer_curves(epochs):
    dataset_size = 50000
    batch_size = 512
    steps_per_epoch = math.ceil(50000/512)

    x = [0, 1, 100, 500] + list(range(1000, steps_per_epoch * epochs, 1000)) + [steps_per_epoch * epochs]

    rdp_all = [get_rdp_eps(sigma=1, real_batch_size=batch_size, steps=step) for step in x]
    rdp_eps = [x[0] for x in rdp_all]
    rdp_alpha = [x[1] for x in rdp_all]

    gamma = np.log(2*len(alphas)/1e-6)/(np.atleast_1d(alphas)-1)
    odom_all = [get_odom_eps(sigma=1, real_batch_size=batch_size, steps=step, gamma=gamma) for step in x]
    odom_eps = [x[0] for x in odom_all]
    odom_alpha = [x[1] for x in odom_all]

    gamma = .5
    odom2_all = [get_odom_eps2(sigma=1, real_batch_size=batch_size, steps=step, gamma=gamma) for step in x]
    odom2_eps = [x[0] for x in odom2_all]
    odom2_alpha = [x[1] for x in odom2_all]

    plt.figure(figsize=(6, 3.), tight_layout=True)
    plt.plot(
        x, rdp_eps,
        "-", color=cycle_colors[0],
        linewidth=linewidth, markersize=markersize,
        label=r"$\epsilon_{RDP}$",
    )
    plt.plot(
        x, odom_eps,
        "-.", color=cycle_colors[1],
        linewidth=linewidth, markersize=markersize,
        label=r"Odometer",
    )
    plt.plot(
        x, odom2_eps,
        "-.", color=cycle_colors[2],
        linewidth=linewidth, markersize=markersize,
        label=r"Odometer 2",
    )
    plt.savefig("rdp_curve_odometer.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

    plt.figure(figsize=(6, 3.), tight_layout=True)
    plt.plot(
        x, rdp_alpha,
        "-", color=cycle_colors[0],
        linewidth=linewidth, markersize=markersize,
        label=r"$\epsilon_{RDP}$",
    )
    plt.plot(
        x, odom_alpha,
        "-.", color=cycle_colors[1],
        linewidth=linewidth, markersize=markersize,
        label=r"Odometer",
    )
    plt.plot(
        x, odom2_alpha,
        "-.", color=cycle_colors[2],
        linewidth=linewidth, markersize=markersize,
        label=r"Odometer 2",
    )
    plt.legend()
    plt.savefig("alpha_curve_odometer.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def plot_finetune_results(path, keys, prefix=""):
    with open(path) as f:
        data = json.loads(f.read())

    plt.figure(figsize=(3, 3.), tight_layout=True)
    plt.ylim(bottom=0.55, top=0.90)

    for i, key in enumerate(keys):
        mean_acc = None
        n = 0
        min_acc = None
        max_acc = None
        for k, v in data.items():
            if key in k:
                acc = np.atleast_1d(v['test_acc'])
                if mean_acc is None:
                    n = 1
                    mean_acc = min_acc = max_acc = acc
                elif len(mean_acc) == len(acc):
                    n += 1
                    mean_acc = acc + mean_acc
                    min_acc = np.minimum(acc, min_acc)
                    max_acc = np.maximum(acc, max_acc)
        mean_acc = mean_acc / n

        x = list(range(len(mean_acc)))
        plt.plot(x, mean_acc, 'k', color=cycle_colors[3+i], label=key_names[key])
        plt.fill_between(x, min_acc, max_acc,
                    alpha=.5, edgecolor=cycle_colors[3+i], facecolor=cycle_colors[3+i],
                        linewidth=0)

    plt.legend(loc="lower right")
    plt.xlabel('Epochs')
    plt.ylabel('Test Set Accuracy')
    plt.savefig(prefix + "fine_tuning_acc.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def plot_adapt_results(path, keys):
    with open(path) as f:
        data = json.loads(f.read())

    x_max = 0
    plt.figure(figsize=(3, 3.), tight_layout=True)
    for i, key in enumerate(keys):
        plt.plot(np.NaN, np.NaN, linestyle=line_styles[key], color=cycle_colors[i], label=key_names[key])
        min_acc = float('+inf')
        final_acc = 0
        max_acc = 0
        n = 0
        for k, v in data.items():
            if key in k:
                acc = v['test_acc']
                final_acc += acc[-1]
                min_acc = min(acc[-1], min_acc)
                max_acc = max(acc[-1], max_acc)
                n += 1
                x_max = max(x_max, len(acc))
                plt.plot(
                    range(len(acc)), acc, linestyle=line_styles[key], color=cycle_colors[i], alpha=.75, linewidth=linewidth,
                )

        print(f"Filter adapt: {key} avg. accuracy={final_acc/n:.4f} min={min_acc:.4f} max={max_acc:.4f}")

    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Test Set Accuracy')
    plt.xticks(np.arange(0, x_max, step=100))
    plt.savefig("adapt_acc.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

    plt.figure(figsize=(3, 3.), tight_layout=True)
    for i, key in enumerate(keys):
        plt.plot(np.NaN, np.NaN, 'k', linestyle=line_styles[key], color=cycle_colors[i], label=key_names[key])
        for k, v in data.items():
            if key in k:
                batch_size = v['batch_size']
                steps = [math.ceil(50000/bs) for bs in batch_size]
                sigma = v['sigma']
                epsilons = []
                rdp = None
                for j in range(len(steps)):
                    eps, alpha, rdp = get_rdp_eps(
                        sigma=sigma[j], real_batch_size=batch_size[j], steps=steps[j], rdp_init=rdp
                    )
                    epsilons.append(eps)
                plt.plot(
                    range(len(epsilons)), epsilons, 'k', linestyle=line_styles[key], color=cycle_colors[i], alpha=.75, linewidth=linewidth,
                )

    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Budget Consumed')
    plt.xticks(np.arange(0, x_max, step=100))
    plt.savefig("adapt_eps.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def plot_adapt_noise_odom_eps(path, key, i):
    with open(path) as f:
        data = json.loads(f.read())

    k = 'adapt_noise'
    plt.figure(figsize=(3, 3.), tight_layout=True)

    for key in ['baseline_0', key]:
        gamma = 6/(np.atleast_1d(alphas)-1)
        batch_size = data[key]['batch_size']
        steps = [math.ceil(50000/bs) for bs in batch_size]
        sigma = data[key]['sigma']
        epsilons = []
        rdp_epsilons = []
        rdp = None
        for j in range(len(steps)):
            eps, alpha, _ = get_rdp_eps(
                sigma=sigma[j], real_batch_size=batch_size[j], steps=steps[j], rdp_init=rdp
            )
            epsilons.append(eps)
            eps, alpha, rdp = get_odom_eps(
                sigma=sigma[j], real_batch_size=batch_size[j], steps=steps[j], rdp_init=rdp, gamma=gamma
            )
            rdp_epsilons.append(eps)

        label = "Baseline RDP" if key == 'baseline_0' else f"{key_names[k]} RDP"
        l = 0 if key == 'baseline_0' else i
        plt.plot(
            range(len(epsilons)), epsilons, 'k', linestyle=line_styles[k], color=cycle_colors[l], alpha=.75, linewidth=linewidth, label=label,
        )
    x_max = len(epsilons)
    plt.plot(
        range(len(rdp_epsilons)), rdp_epsilons, 'k', linestyle=(0, (5, 1)), color=cycle_colors[i], alpha=.75, linewidth=linewidth, label=f"{key_names[k]} Odometer",
    )
    x_max = max(x_max, len(rdp_epsilons))

    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Budget Consumed')
    plt.xticks(np.arange(0, x_max, step=100))
    plt.savefig("adapt_eps_odom.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def plot_adapt_batch_example(path, key, i):
    with open(path) as f:
        data = json.loads(f.read())
    plt.figure(figsize=(3, 3.), tight_layout=True)
    plt.ylim(bottom=128, top=550)

    y = data['baseline_0']['batch_size']
    x_max = len(y)
    k = 'baseline'
    plt.plot(
        range(len(y)), y, 'k', linestyle=line_styles[k], color=cycle_colors[0], alpha=.75, linewidth=linewidth,
        label=key_names[k]
    )
    y = data[key]['batch_size']
    x_max = max(x_max, len(y))
    k = 'adapt_batch'
    plt.plot(
        range(len(y)), y, 'k', linestyle=line_styles[k], color=cycle_colors[i], alpha=.75, linewidth=linewidth,
        label=key_names[k]
    )

    plt.legend(loc='lower center', ncol=1)
    plt.xticks(np.arange(0, x_max, step=100))
    plt.yticks(np.arange(128, 550, step=128))
    plt.xlabel('Epochs')
    plt.ylabel('Batch Size')
    plt.savefig("adapt_batch.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def plot_adapt_noise_example(path, key, i):
    with open(path) as f:
        data = json.loads(f.read())
    plt.figure(figsize=(3, 3.), tight_layout=True)
    plt.ylim(bottom=0.9, top=1.85)

    y = data['baseline_0']['sigma']
    x_max = len(y)
    k = 'baseline'
    plt.plot(
        range(len(y)), y, 'k', linestyle=line_styles[k], color=cycle_colors[0], alpha=.75, linewidth=linewidth,
        label=key_names[k]
    )
    y = data[key]['sigma']
    x_max = max(x_max, len(y))
    k = 'adapt_noise'
    plt.plot(
        range(len(y)), y, 'k', linestyle=line_styles[k], color=cycle_colors[i], alpha=.75, linewidth=linewidth,
        label=key_names[k]
    )

    plt.legend(loc='lower right', ncol=1)
    plt.xticks(np.arange(0, x_max, step=100))
    plt.xlabel('Epochs')
    plt.ylabel(r'DP Noise $\sigma$')
    plt.savefig("adapt_noise.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

def plot_finetuning_odometer_curves(path, k_list, key_list, i_list, include_baseline=True, prefix=""):
    batch_size = 512
    steps_per_epoch = math.ceil(dataset_size/batch_size)

    with open(path) as f:
        data = json.loads(f.read())

    epochs = len(data[key_list[0]]['batch_size'])
    gamma = 6/(np.atleast_1d(alphas)-1)

    plt.figure(figsize=(3, 3.), tight_layout=True)
    plt.ylim(bottom=0., top=9.5)
    # Baseline
    if include_baseline:
        k='baseline'
        steps_per_epochs = math.ceil(dataset_size/batch_size)
        rdp_all = [get_rdp_eps(sigma=1, real_batch_size=batch_size, steps=e*steps_per_epochs) for e in range(epochs+1)]
        rdp_eps = [x[0] for x in rdp_all]
        plt.plot(
            range(epochs+1), rdp_eps, 'k', linestyle=line_styles[k], color=cycle_colors[0], alpha=.75,
            linewidth=linewidth, label="Non-adaptive RDP",
        )

    # Finetuning RDP and Odometer
    for i, key in enumerate(key_list):
        k = k_list[i]
        batch_size = data[key]['batch_size']
        steps = [math.ceil(dataset_size/bs) for bs in batch_size]
        sigma = data[key]['sigma']
        epsilons = []
        rdp_epsilons = []
        rdp = None
        eps, alpha, _ = get_rdp_eps(sigma=1, real_batch_size=512, steps=0, rdp_init=rdp)
        epsilons.append(eps)
        eps, alpha, rdp = get_odom_eps(sigma=1, real_batch_size=512, steps=0, rdp_init=rdp, gamma=gamma)
        rdp_epsilons.append(eps)
        for j in range(len(steps)):
            eps, alpha, _ = get_rdp_eps(
                sigma=sigma[j], real_batch_size=batch_size[j], steps=steps[j], rdp_init=rdp
            )
            epsilons.append(eps)
            eps, alpha, rdp = get_odom_eps(
                sigma=sigma[j], real_batch_size=batch_size[j], steps=steps[j], rdp_init=rdp, gamma=gamma
            )
            rdp_epsilons.append(eps)

        plt.plot(
            range(len(epsilons)), epsilons, 'k', linestyle=line_styles[k], color=cycle_colors[i_list[i]+fine_tuning_offset], alpha=.75, linewidth=linewidth, label=f"RDP",
        )
        x_max = len(epsilons)
        plt.plot(
            range(len(rdp_epsilons)), rdp_epsilons, 'k', linestyle=(0, (5,1)), color=cycle_colors[i_list[i]+fine_tuning_offset], alpha=.75, linewidth=linewidth, label=f"Odometer",
        )
        x_max = max(x_max, len(rdp_epsilons))

    plt.legend(loc='upper left')
    plt.xlabel('Epochs')
    plt.ylabel('Budget Consumed')
    plt.savefig(prefix + "finetuning_rdp_curve_odometer.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

    plt.figure(figsize=(3, 3.), tight_layout=True)
    plt.ylim(bottom=0., top=32)
    batch_size = 512
    # Baseline
    if include_baseline:
        k='baseline'
        steps_per_epochs = math.ceil(dataset_size/batch_size)
        rdp_all = [get_rdp_eps(sigma=1, real_batch_size=batch_size, steps=e*steps_per_epochs) for e in range(epochs+1)]
        rdp_alphas = [x[1] for x in rdp_all]
        plt.plot(
            range(epochs+1), rdp_alphas, 'k', linestyle=line_styles[k], color=cycle_colors[0], alpha=.75,
            linewidth=linewidth, label="Non-adaptive RDP",
        )

    # Finetuning RDP and Odometer
    for i, key in enumerate(key_list):
        k = k_list[i]
        batch_size = data[key]['batch_size']
        steps = [math.ceil(dataset_size/bs) for bs in batch_size]
        sigma = data[key]['sigma']
        _alphas = []
        _rdp_alphas = []
        rdp = None
        eps, alpha, _ = get_rdp_eps(sigma=1, real_batch_size=512, steps=0, rdp_init=rdp)
        _alphas.append(alpha)
        eps, alpha, rdp = get_odom_eps(sigma=1, real_batch_size=512, steps=0, rdp_init=rdp, gamma=gamma)
        _rdp_alphas.append(alpha)
        for j in range(len(steps)):
            eps, alpha, _ = get_rdp_eps(
                sigma=sigma[j], real_batch_size=batch_size[j], steps=steps[j], rdp_init=rdp
            )
            _alphas.append(alpha)
            eps, alpha, rdp = get_odom_eps(
                sigma=sigma[j], real_batch_size=batch_size[j], steps=steps[j], rdp_init=rdp, gamma=gamma
            )
            _rdp_alphas.append(alpha)

        plt.plot(
            range(len(_alphas)), _alphas, 'k', linestyle=line_styles[k], color=cycle_colors[i_list[i]+fine_tuning_offset], alpha=.75, linewidth=linewidth, label=f"Adaptive RDP",
        )
        x_max = len(_alphas)
        plt.plot(
            range(len(_rdp_alphas)), _rdp_alphas, 'k', linestyle=(0, (5,1)), color=cycle_colors[i_list[i]+fine_tuning_offset], alpha=.75, linewidth=linewidth, label=f"Adaptive Odometer",
        )
        x_max = max(x_max, len(_rdp_alphas))

    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel(r'Best $\alpha$')
    plt.savefig(prefix + "finetuning_alpha_curve_odometer.pdf", bbox_inches='tight', pad_inches=0.0)
    plt.clf()

fine_tuning_offset = 3
odometer_offset = 5
key_names = {
    'baseline': "Baseline",
    'adapt_noise': r"Adapt $\sigma$",
    'adapt_batch': "Adapt batch size",
    'fine_tune_last': 'Fine-tune last',
    'fine_tune_all': 'Fine-tune all',
}
line_styles = {
    'baseline': "solid",
    'adapt_noise': "solid",
    'adapt_batch': "solid",
    'fine_tune_last': 'solid',
    'fine_tune_all': 'solid',
}

if __name__ == "__main__":
    plot_adapt_results("exp_results/adapt_exp.json", keys=['baseline', 'adapt_noise', 'adapt_batch'])
    plot_adapt_batch_example("exp_results/adapt_exp.json", key='adapt_batch_1', i=2)
    plot_adapt_noise_example("exp_results/adapt_exp.json", key='adapt_noise_3', i=1)
    plot_adapt_noise_odom_eps("exp_results/adapt_exp.json", key='adapt_noise_3', i=1)
    #
    plot_finetune_results("exp_results/fine_tune_exp.json", keys=['fine_tune_last', 'fine_tune_all'], prefix="filter_")
    plot_finetune_results("exp_results/fine_tune_adapt_exp.json", keys=['fine_tune_last', 'fine_tune_all'], prefix="adapt_")
    plot_finetuning_odometer_curves("exp_results/fine_tune_exp.json", include_baseline=False, k_list=['fine_tune_last'], key_list=['fine_tune_last_0'], i_list=[0], prefix="early_stop_")
    plot_finetuning_odometer_curves("exp_results/fine_tune_adapt_exp.json", k_list=['fine_tune_last'], key_list=['fine_tune_last_0'], i_list=[0], prefix="adaptive_")

