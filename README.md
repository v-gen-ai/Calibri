# ScaleGuidance

## Installation

Framework was tested with python = 3.10

```
pip install -r requirements.txt
```

## Supported Models

- **FLUX.1-dev**: `black-forest-labs/FLUX.1-dev`
- **Stable Diffusion 3**: `stabilityai/stable-diffusion-3.5-medium`, `stabilityai/stable-diffusion-3.5-large`
- **Qwen-Image**: `Qwen/Qwen-Image` 

## Quick Start

### FLUX/SD3
```
accelerate launch --num_processes 2 scripts/train.py --config configs/scaleguidance.py:cmaes_hpsv3_2models
```

### Qwen-Image
```
# Full training
accelerate launch --num_processes 2 scripts/train.py --config configs/scaleguidance.py:cmaes_qwen_hpsv3
```

## Configuration

You can find different configs at `configs/scaleguidance.py`:

- `cmaes_hpsv3_2models`: FLUX with HPSv3 scoring
- `cmaes_qwen_hpsv3`: Qwen-Image with HPSv3 scoring
- `cmaes_qwen_hpsv3_2models`: Two Qwen-Image models

## Monitoring

### TensorBoard
```
tensorboard --logdir=<exp_logdir>
```


## Documentation

- **Qwen-Image Integration**: [QWEN_INTEGRATION.md](QWEN_INTEGRATION.md)
