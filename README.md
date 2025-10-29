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
accelerate launch --num_processes 2 --config_file configs/2gpu.yaml scripts/train.py --config configs/scaleguidance.py:cmaes_hpsv3_2models
```

### Qwen-Image
```
# Test integration
python scripts/quick_test_qwen.py

# Full training
accelerate launch --num_processes 2 --config_file configs/2gpu.yaml scripts/train.py --config configs/scaleguidance.py:cmaes_qwen_hpsv3
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

### SwanLab (Cloud Dashboard)
SwanLab provides cloud-based experiment tracking with real-time monitoring.

**Available SwanLab configurations:**
- `cmaes_qwen_hpsv3_swanlab`: Qwen-Image with SwanLab
- `cmaes_hpsv3_2models_swanlab`: FLUX with SwanLab


**Run with SwanLab:**
```bash
accelerate launch --num_processes 2 --config_file configs/2gpu.yaml scripts/train.py --config configs/scaleguidance.py:cmaes_qwen_hpsv3_swanlab
```

## Documentation

- **Qwen-Image Integration**: [QWEN_INTEGRATION.md](QWEN_INTEGRATION.md)
