# Qwen-Image Integration with ScaleGuidance

This document describes how to use Qwen-Image with the ScaleGuidance framework.

## Overview

Qwen-Image has been integrated into ScaleGuidance as a new model type alongside FLUX and Stable Diffusion 3. The integration adds trainable scaling factors to the transformer blocks of Qwen-Image, allowing for optimization via CMA-ES.

## Architecture

### Qwen-Image Transformer Structure
- **60 transformer blocks** (configurable)
- **4 scaling parameters per block**:
  - `img_attn`: Image attention gate scaling
  - `txt_attn`: Text attention gate scaling  
  - `img_mlp`: Image MLP gate scaling
  - `txt_mlp`: Text MLP gate scaling

### Total Parameters
- **240 parameters** for single model (60 blocks × 4 parameters)
- **480 parameters** for 2 models (2 × 240)
- Plus model blending weights

## Usage

### 1. Quick Test
```bash
cd scaleguidance
python scripts/quick_test_qwen.py
```

### 2. Full Integration Test
```bash
python scripts/test_qwen.py --config configs/scaleguidance.py:cmaes_qwen_hpsv3
```

### 3. Training with CMA-ES

#### Single Model
```bash
accelerate launch --num_processes 2 --config_file configs/2gpu.yaml scripts/train.py --config configs/scaleguidance.py:cmaes_qwen_hpsv3
```

#### Two Models
```bash
accelerate launch --num_processes 2 --config_file configs/2gpu.yaml scripts/train.py --config configs/scaleguidance.py:cmaes_qwen_hpsv3_2models
```

## Configuration

### Available Configurations
- `cmaes_qwen_hpsv3`: Single Qwen model with HPSv3 scoring
- `cmaes_qwen_hpsv3_2models`: Two Qwen models with HPSv3 scoring

### Key Parameters
```python
cfg.model.model_name = "Qwen/Qwen-Image"
cfg.model.dtype = "bf16"
cfg.gen.guidance_scale = 4.0  # CFG scale for Qwen
cfg.scaleguidance.num_models = 1  # or 2
cfg.reward_fn = {"hpsv3_remote": 1.0}
```

## HPSv3 Remote Server

The integration uses HPSv3 remote scoring. Make sure your HPSv3 server is running:

```bash
# Start HPSv3 server (adjust URL as needed)
# Default URL: http://127.0.0.1:18087
```

## File Structure

```
scaleguidance/
├── src/models/
│   ├── qwen_sg.py          # SGQwenPipeline implementation
│   └── __init__.py         # Updated with Qwen support
├── configs/
│   └── scaleguidance.py    # Qwen configurations added
├── scripts/
│   ├── test_qwen.py        # Full integration test
│   └── quick_test_qwen.py  # Quick functionality test
└── QWEN_INTEGRATION.md     # This file
```
