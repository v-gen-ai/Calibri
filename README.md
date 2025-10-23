# ScaleGuidance

## Installation

Framework was tested with python = 3.10

```
pip install -r requirements.txt
```

To run the experiment:
```
accelerate launch --num_processes 2 --config_file configs/2gpu.yaml scripts/train.py --config configs/scaleguidance.py:cmaes_hpsv3_2models
```

You can find different configs at configs/gatescale.py

Plots:
```
tensorboard --logdir=<exp_logdir>
```