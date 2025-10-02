import os
import json
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import cma

from src.utils.logging_tb import log_scalars, log_hist_alphas, log_images


def _get_device(cfg) -> torch.device:
    return torch.device(getattr(cfg, "device", "cuda"))


def _gen_params(cfg) -> Dict:
    return {
        "num_inference_steps": int(cfg.gen.num_inference_steps),
        "guidance_scale": float(cfg.gen.guidance_scale),
        "height": int(cfg.gen.image_size),
        "width": int(cfg.gen.image_size),
    }


def _transformer_dims(pipeline) -> Tuple[int, int, int]:
    t = pipeline.pipeline.transformer
    n_double = len(t.transformer_blocks)
    n_single = len(t.single_transformer_blocks)
    n_models = len(t.models_scales)
    return n_double, n_single, n_models


def _flatten_from_transformer(pipeline) -> np.ndarray:
    t = pipeline.pipeline.transformer
    n_double, n_single, n_models = _transformer_dims(pipeline)
    vec = []
    # doubles (attn, mlp)
    for m in range(n_models):
        for b in range(n_double):
            w_attn, w_mlp = t.transformer_gate_scales[m][b]
            vec.append(float(w_attn.detach().cpu().item()))
            vec.append(float(w_mlp.detach().cpu().item()))
    # singles
    for m in range(n_models):
        for b in range(n_single):
            w = t.single_gate_scales[m][b]
            vec.append(float(w.detach().cpu().item()))
    # models_scales
    ms = t.models_scales.detach().cpu().to(torch.float64).numpy().astype(np.float64).tolist()
    vec.extend(ms)
    return np.asarray(vec, dtype=np.float64)


def _vector_shapes(pipeline) -> Dict[str, int]:
    n_double, n_single, n_models = _transformer_dims(pipeline)
    doubles = n_models * n_double * 2
    singles = n_models * n_single
    models = n_models
    return {"doubles": doubles, "singles": singles, "models": models, "total": doubles + singles + models,
            "n_double": n_double, "n_single": n_single, "n_models": n_models}


def _unflatten_to_gs(x: np.ndarray, shapes: Dict[str, int]):
    d_doubles = shapes["doubles"]
    d_singles = shapes["singles"]
    n_double = shapes["n_double"]
    n_single = shapes["n_single"]
    n_models = shapes["n_models"]

    doubles = x[:d_doubles]
    singles = x[d_doubles:d_doubles + d_singles]
    models = x[d_doubles + d_singles:]

    # build per-model
    scales_double = []
    scales_single = []

    # doubles per model: n_double * 2
    per_model_doubles = n_double * 2
    for m in range(n_models):
        md = []
        start = m * per_model_doubles
        for b in range(n_double):
            a = doubles[start + 2 * b + 0]
            c = doubles[start + 2 * b + 1]
            md.append((float(a), float(c)))
        scales_double.append(md)

    # singles per model: n_single
    for m in range(n_models):
        ms = []
        start = m * n_single
        for b in range(n_single):
            ms.append(float(singles[start + b]))
        scales_single.append(ms)

    models_scales = [float(v) for v in models.tolist()]
    return scales_double, scales_single, models_scales


def _mean_score(scores) -> float:
    if scores is None:
        return -np.inf
    if isinstance(scores, (list, tuple)):
        if len(scores) == 0:
            return -np.inf
        return float(np.mean([float(s) for s in scores]))
    if torch.is_tensor(scores):
        if scores.numel() == 0:
            return -np.inf
        return float(scores.float().mean().item())
    return float(scores)


def _call_reward(fn, images, prompts):
    # Унифицируем контракт: всегда возвращаем только список/массив числовых scores
    try:
        out = fn(images, prompts, None)
    except TypeError:
        out = fn(images, prompts)
    return list(out[0])


class CMAESTrainer:
    def __init__(self, cfg, pipeline, reward_fn, eval_reward_fn, writer, train_loader, val_loader):
        self.cfg = cfg
        self.pipeline = pipeline  # GSFluxPipeline
        self.reward_fn = reward_fn
        self.eval_reward_fn = eval_reward_fn
        self.writer = writer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = _get_device(cfg)
        self.gen_params = _gen_params(cfg)

        self.shapes = _vector_shapes(self.pipeline)
        self.dim = self.shapes["total"]

        # Initial point from current transformer state
        x0 = _flatten_from_transformer(self.pipeline)

        # CMA-ES options
        opts = {
            "seed": int(cfg.experiment.seed) if getattr(cfg.experiment, "seed", None) is not None else None,
        }
        if getattr(cfg.optimize, "population_size", None):
            opts["popsize"] = int(cfg.optimize.population_size)

        bounds_low: List[float] = []
        bounds_high: List[float] = []
        has_bounds = False

        if getattr(cfg.gatescale, "blocks_bound_low", None) is not None:
            blocks_bound_low = cfg.optimize.blocks_bound_low
            blocks_bound_high = cfg.optimize.blocks_bound_high
            models_bound_low = cfg.optimize.models_bound_low
            models_bound_high = cfg.optimize.models_bound_high
            has_bounds = True

        if has_bounds:
            bounds_low.extend([float(blocks_bound_low)] * self.shapes["doubles"])
            bounds_high.extend([float(blocks_bound_high)] * self.shapes["doubles"])

            bounds_low.extend([float(blocks_bound_low)] * self.shapes["singles"])
            bounds_high.extend([float(blocks_bound_high)] * self.shapes["singles"])

            bounds_low.extend([float(models_bound_low)] * self.shapes["models"])
            bounds_high.extend([float(models_bound_high)] * self.shapes["models"])

            opts["bounds"] = [bounds_low, bounds_high]

        self.es = cma.CMAEvolutionStrategy(
            x0=x0,
            sigma0=float(cfg.optimize.initial_sigma),
            inopts=opts,
        )

        # state
        self._train_iter = iter(self.train_loader)
        self.best_solution = x0.copy()
        self.best_train = -np.inf
        self.best_val = -np.inf
        self.patience = 0

        # history
        self.hist_train_best: List[float] = []
        self.hist_train_mean: List[float] = []
        self.hist_val: List[Optional[float]] = []
        self.hist_sigma: List[float] = []
        self.hist_models_scales: List[List[float]] = []

    def _next_batch(self) -> List[str]:
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        return list(batch)

    def _apply_x_via_pipeline(self, x: np.ndarray):
        scales_double, scales_single, models_scales = _unflatten_to_gs(x, self.shapes)
        self.pipeline.modify_gatescale(
            num_models=self.shapes["n_models"],
            scales_double=scales_double,
            scales_single=scales_single,
            models_scales=models_scales,
        )

    def _gen_images(self, prompts: List[str], seed: int):
        # For fair comparison within a generation, fix generator
        gen_device = "cpu"  # reproducibility: CPU generator recommended by diffusers
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))
        out = self.pipeline(
            prompts,
            num_inference_steps=self.gen_params["num_inference_steps"],
            guidance_scale=self.gen_params["guidance_scale"],
            height=self.gen_params["height"],
            width=self.gen_params["width"],
            generator=generator,
        )
        return out.images

    def _eval_candidate_on_batch(self, x: np.ndarray, prompts: List[str], seed: int) -> float:
        self._apply_x_via_pipeline(x)
        images = self._gen_images(prompts, seed)
        scores = _call_reward(self.reward_fn, images, prompts)
        return _mean_score(scores)

    def _eval_validation(self, x: np.ndarray, seed: int = 1234) -> float:
        self._apply_x_via_pipeline(x)
        total = 0.0
        count = 0
        gen_device = "cpu"
        for prompts in self.val_loader:
            generator = torch.Generator(device=gen_device).manual_seed(int(seed))
            out = self.pipeline(
                list(prompts),
                num_inference_steps=self.gen_params["num_inference_steps"],
                guidance_scale=self.gen_params["guidance_scale"],
                height=self.gen_params["height"],
                width=self.gen_params["width"],
                generator=generator,
            )
            images = out.images
            scores = _call_reward(self.eval_reward_fn, images, list(prompts))
            if isinstance(scores, (list, tuple)):
                total += float(np.sum([float(s) for s in scores]))
                count += len(scores)
            elif torch.is_tensor(scores):
                total += float(scores.float().sum().item())
                count += int(scores.numel())
            else:
                total += float(scores)
                count += 1
        return float(total / max(1, count))

    def _log_step(self, step: int, best_fit: float, mean_fit: float, sigma: float, best_x: np.ndarray):
        scalars = {
            "train/best": float(best_fit),
            "train/mean": float(mean_fit),
            "cmaes/sigma": float(sigma),
        }
        log_scalars(self.writer, scalars, step)

        d_doubles = self.shapes["doubles"]
        d_singles = self.shapes["singles"]
        doubles = best_x[:d_doubles]
        singles = best_x[d_doubles:d_doubles + d_singles]
        models = best_x[d_doubles + d_singles:]

        attn = doubles[0::2]
        mlp = doubles[1::2]

        alpha_dict = {
            "double_attn": np.array(attn, dtype=np.float32),
            "double_mlp": np.array(mlp, dtype=np.float32),
            "single": np.array(singles, dtype=np.float32),
            "models_scales": np.array(models, dtype=np.float32),
        }
        log_hist_alphas(self.writer, alpha_dict, step, prefix="alphas/")

        self.hist_sigma.append(float(sigma))
        self.hist_models_scales.append(models.tolist())

    def _maybe_log_images(self, step: int, prompts: List[str], x: np.ndarray, tag: str = "samples/train"):
        if not getattr(self.cfg.experiment, "save_images", False):
            return
        self._apply_x_via_pipeline(x)
        show_seed = int(getattr(self.cfg.experiment, "seed", 0)) + step + 777
        images = self._gen_images(prompts, show_seed)
        log_images(self.writer, tag, images, step)

    def _checkpoint_json(self, step: int, best_fit: float, mean_fit: float, val_fit: Optional[float], best_x: np.ndarray):
        if not getattr(self.cfg.experiment, "save_json", False):
            return
        run_dir = getattr(self.writer, "log_dir", None) or os.path.join(self.cfg.experiment.log_dir, self.cfg.experiment.name)
        os.makedirs(run_dir, exist_ok=True)
        payload = {
            "step": int(step),
            "train_best": float(best_fit),
            "train_mean": float(mean_fit),
            "val": None if val_fit is None else float(val_fit),
            "solution": [float(v) for v in best_x.tolist()],
            "shapes": self.shapes,
            "timestamp": int(time.time()),
        }
        with open(os.path.join(run_dir, f"cmaes_step_{step:04d}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def train(self) -> Tuple[np.ndarray, float, float]:
        generation = 0
        max_gens = int(self.cfg.optimize.max_generations)
        base_seed = int(getattr(self.cfg.experiment, "seed", 0))

        while not self.es.stop() and generation < max_gens:
            step = generation + 1

            # one shared train batch per generation
            batch_prompts = self._next_batch()

            # evaluate candidates on the same batch
            candidates = self.es.ask()
            fitnesses = []
            for x in candidates:
                score = self._eval_candidate_on_batch(x, batch_prompts, seed=base_seed + generation)
                fitnesses.append(-float(score))  # CMA-ES minimizes

            self.es.tell(candidates, fitnesses)

            # generation metrics
            best_idx = int(np.argmin(fitnesses))
            best_x = np.asarray(candidates[best_idx], dtype=np.float64)
            best_fit = -float(fitnesses[best_idx])
            mean_fit = -float(np.mean(fitnesses))

            # track best
            if best_fit > self.best_train:
                self.best_train = best_fit
                self.best_solution = best_x.copy()

            self.hist_train_best.append(best_fit)
            self.hist_train_mean.append(mean_fit)

            # logging
            self._log_step(step, best_fit, mean_fit, float(self.es.sigma), best_x)
            self._maybe_log_images(step, batch_prompts, best_x)

            # validation schedule
            do_val = int(self.cfg.optimize.val_every_steps) > 0 and (step % int(self.cfg.optimize.val_every_steps) == 0 or step == max_gens)
            if do_val:
                val_score = self._eval_validation(best_x, seed=1234)
                self.hist_val.append(val_score)
                log_scalars(self.writer, {"val/mean": float(val_score)}, step)

                train_val_gap = best_fit - float(val_score)
                if val_score > self.best_val:
                    self.best_val = float(val_score)
                    self.patience = 0
                else:
                    self.patience += 1

                # if self.patience >= int(self.cfg.optimize.early_stopping_patience):
                #     self._checkpoint_json(step, best_fit, mean_fit, val_score, best_x)
                #     break

                # if train_val_gap > float(self.cfg.optimize.overfitting_threshold) and self.patience >= int(self.cfg.optimize.early_stopping_patience):
                #     self._checkpoint_json(step, best_fit, mean_fit, val_score, best_x)
                #     break
            else:
                self.hist_val.append(None)

            # checkpoint
            self._checkpoint_json(step, best_fit, mean_fit, self.hist_val[-1], best_x)

            generation += 1

        # final checkpoint
        last_mean = float(self.hist_train_mean[-1]) if self.hist_train_mean else float("-inf")
        last_val = self.best_val if (self.hist_val and self.hist_val[-1] is not None) else None
        self._checkpoint_json(generation, self.best_train, last_mean, last_val, self.best_solution)

        return self.best_solution, float(self.best_train), float(self.best_val if self.best_val != -np.inf else np.nan)
