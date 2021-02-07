# adaptive_rdp

## Requirements

- CUDA 10.2+ drivers
- python3.7
- torch torchvision opacus tqdm matplotlib pandas requests torchcsprng
    tensorboard seaborn

## Plot results from included log files

    python plots.py

Or:

    make plot

## Re-run paper's experiments

    python exp_adaptive.py
    python exp_fine_tuning.py

Or:

    make exp

## Train a single model

For tensorbord logs:

    tensorboard --logdir=/tmp/stat

Run e.g.:

    # Adaptive training:
	  python main.py -b 128 -na 4 --sigma 1 --filter 10 --adaptive noise --adaptive-metric train_acc --adaptive-strategy updown --maximum-factor 1. --adaptation-additive .1 --adaptation-period 10 --lr 0.05 --epochs 300 --wd 0 --model-type ResNet9 --log-dir ResNet9_adaptive

    # Fine-tuning:
    python main.py -b 64 -na 8 --sigma 2 --odometer --adaptive noise --adaptive-metric train_acc --adaptive-strategy down --maximum-factor 1. --adaptation-additive .1 --adaptation-period 1 --enable-bn --lr 0.01 --epochs 20 --wd 0 --model-type ResNet18 --fine-tune last --log-dir ResNet18_finetuning
