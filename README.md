# ScaleGuidance

## Installation

Framework was tested with python = 3.10

```
pip install -r requirements.txt
```

To run the experiment:
```
python scripts/train.py --config configs/scaleguidance.py:cmaes_image_reward
```

You can find different configs at configs/gatescale.py

Plots:
```
tensorboard --logdir=<exp_logdir>
```