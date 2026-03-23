<h1 align="center"> Calibri:<br>Enhancing Diffusion Transformers via Parameter-Efficient Calibration </h1>

<div align="center">
  <a href='https://arxiv.org/abs/2505.05470'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
  <a href='https://gongyeliu.github.io/Flow-GRPO/'><img src='https://img.shields.io/badge/Visualization-green?logo=github'></a> &nbsp;
  <a href="https://github.com/yifan123/flow_grpo"><img src="https://img.shields.io/badge/Code-9E95B7?logo=github"></a> &nbsp; 
  <a href='https://huggingface.co/collections/jieliu/sd35m-flowgrpo-68298ec27a27af64b0654120'><img src='https://img.shields.io/badge/Model-blue?logo=huggingface'></a> &nbsp; 
  <a href='https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO'><img src='https://img.shields.io/badge/Demo-blue?logo=huggingface'></a> &nbsp;
</div>

<br>

> **Calibri** is a parameter-efficient approach that optimally calibrates Diffusion Transformer (DiT) components to elevate generative quality. By framing DiT calibration as a black-box reward optimization problem solved using the CMA-ES evolutionary algorithm, Calibri modifies just **~100 parameters**. This lightweight calibration not only consistently improves generation quality across various models but also significantly reduces the required inference steps (NFE) while maintaining high-quality outputs.

# 📄 Changelog
<details open>
<summary><strong>2026-03-24</strong></summary>

* Official release of **Calibri** codebase! Code supports CMA-ES calibration for **FLUX**, **Stable Diffusion 3.5**, and **Qwen-Image**.

</details>


# 🤗 Supported Models & Rewards

Calibri optimizes text-to-image models by maximizing human-preference rewards. It currently supports the following DiT architectures and Reward Models:

| Task | Model | NFE with Calibri |
| -------- | -------- | -------- |
| Text-to-Image | [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | **15** |
| Text-to-Image | [stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | **30** |
| Text-to-Image | [stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | **30** |
| Text-to-Image | [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | **30** |


**Supported Reward Models:**
* **[HPSv3](https://github.com/tgxs002/HPSv3)**: Human Preference Score v3.
* **[Q-Align](https://github.com/Q-Future/Q-Align)**: MLLM-based visual quality scoring.
* **[PickScore](https://huggingface.co/yuvalkirstain/PickScore_v1)**: CLIP-based aesthetic scoring model.
* **[ImageReward](https://github.com/THUDM/ImageReward)**: General human preference reward.


# 🚀 Quick start

## Environment Set Up
The framework is build with [uv](https://github.com/astral-sh/uv) — an extremely fast Python package and project manager. Installation guide is at uv [docs](https://docs.astral.sh/uv/getting-started/installation/)

**1. Clone the repository**
```bash
git clone https://github.com/your-username/Calibri.git
cd Calibri
```

**2. Setup environment and install dependencies**
```bash
uv sync
source .venv/bin/activate
```

## Reward Preparation

To train using HPSv3 or Q-Align rewards you need to start the reward servers before running the main training script.

HPSv3 server:

```bash
uv run src/metrics/hpsv3_server.py --device cuda:0
```

Q-Align server:
```bash
uv run src/metrics/qalign_server.py --device cuda:1
```

## Start Training
You can easily start the calibration process using Accelerate. The algorithm utilizes the CMA-ES evolutionary strategy to find the optimal scaling parameters.

```bash
accelerate launch --num_processes 2 scripts/train.py --config configs/calibri.py:cmaes_hpsv3_flux_layer
```

### ⚙️ Hyperparameters & Granularity

Calibri is designed to be highly flexible. You can easily customize the target DiT backbone, reward models, and optimization hyperparameters directly via `configs/calibri.py`.

A core feature of our framework is the ability to define the **search space granularity**. As described in our paper, Calibri supports three distinct levels of granularity for internal-layer calibration, allowing you to balance parameter efficiency and generation quality:

* **Block Scaling**: Uniformly adjusts the outputs of Attention and MLP layers within the same block (~57 parameters).
* **Layer Scaling**: Adjusts individual layers within a block using distinct coefficients (~76 parameters).
* **Gate Scaling**: Specialized calibration for visual and textual tokens processed through distinct gates in MM-DiT architectures (~114 parameters).

### 📈 Monitoring
Track your calibration progress, reward metrics, and generated image samples in real-time with tensorboard:

```bash
tensorboard --logdir=<exp_logdir>
```

# 🤗 Acknowledgements

This repository is based on [diffusers](https://github.com/huggingface/diffusers/), [accelerate](https://github.com/huggingface/accelerate) and [flow_grpo](https://github.com/yifan123/flow_grpo/tree/main).
We thank them for their contributions to the community!!!

# ⭐Citation
If you find Calibri useful for your research or projects, we would greatly appreciate it if you could cite the following paper:

```bibtex
@article{tokhchukov2026calibri,
  title={Calibri: Enhancing Diffusion Transformers via Parameter-Efficient Calibration}, 
  author={Tokhchukov, Danil and Mirzoeva, Aysel and Kuznetsov, Andrey and Sobolev, Konstantin},
  journal={arXiv preprint arXiv:2600.00000},
  year={2026},
}
```
