all:
	python main.py -b 128 -na 4 --sigma 1 --filter 10 --adaptive noise --adaptive-metric train_acc --adaptive-strategy updown --maximum-factor 1. --adaptation-additive .1 --adaptation-period 10 --lr 0.05 --epochs 300 --wd 0 --model-type ResNet9 --log-dir ResNet9_adaptive

exp:
	python exp_adaptive.py
	python exp_fine_tuning.py

plot:
	python plots.py
