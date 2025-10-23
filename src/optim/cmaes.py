import os
import json
import time
from typing import List, Tuple, Dict, Optional
import pickle

import numpy as np
from tqdm import tqdm
import torch
import cma
from PIL import Image
import torch.distributed as dist

from src.utils.logging_tb import log_scalars, log_images, log_scatter
from src.utils.utils import to_pil_list, call_reward, mean_score
from src.data.prompts import get_lines


def _get_device(cfg) -> torch.device:
    return torch.device(getattr(cfg, "device", "cuda"))


def _gen_params(cfg) -> Dict:
    return {
        "num_inference_steps": int(cfg.gen.num_inference_steps),
        "guidance_scale": float(cfg.gen.guidance_scale),
        "height": int(cfg.gen.image_size),
        "width": int(cfg.gen.image_size),
    }

def _iter_chunks(seq, size):
    for i in tqdm(range(0, len(seq), size), desc="bucket iteration during train"):
        yield seq[i:i + size]


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

    models_scales = [float(v) for v in list(models)]
    return scales_double, scales_single, models_scales


class CMAESTrainer:
    def __init__(self, cfg, pipeline, reward_fn, eval_reward_fn, writer, 
                 train_loader, val_loader, logdir=None, accelerator=None):
        self.cfg = cfg
        self.pipeline = pipeline  # SGFluxPipeline
        self.reward_fn = reward_fn
        self.eval_reward_fn = eval_reward_fn
        self.writer = writer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logdir = logdir
        self.accelerator = accelerator

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

        if getattr(cfg.scaleguidance, "blocks_bound_low", None) is not None:
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

        # self.es = cma.CMAEvolutionStrategy(
        #     x0=x0,
        #     sigma0=float(cfg.optimize.initial_sigma),
        #     inopts=opts,
        # )

        # # state
        # self._train_iter = iter(self.train_loader) if self.train_loader is not None else None
        # self.best_solution = x0.copy()
        # self.best_train = -np.inf
        # self.best_val = -np.inf
        # self.patience = 0

        self.generation = 0
        resume_state = getattr(cfg.experiment, "resume_state", None)
        resume_json  = getattr(cfg.experiment, "resume_json",  None)

        if resume_state and os.path.isfile(resume_state):
            with open(resume_state, "rb") as f:
                st = pickle.load(f)
            self.es = st["es"]
            self.generation = int(st.get("generation", 0))
            self.best_solution = np.asarray(st.get("best_solution", x0), dtype=np.float64)
            self.best_train = float(st.get("best_train", -np.inf))
            self.best_val = float(st.get("best_val", -np.inf))
            self.hist_train_best = list(st.get("hist_train_best", []))
            self.hist_train_mean = list(st.get("hist_train_mean", []))
            self.hist_train_best_scores = list(st.get("hist_train_best_scores", []))
            self.hist_val = list(st.get("hist_val", []))
            self.hist_sigma = list(st.get("hist_sigma", []))
            self.hist_models_scales = list(st.get("hist_models_scales", []))
            # RNG
            if st.get("np_rng_state", None) is not None:
                np.random.set_state(st["np_rng_state"])
            if st.get("torch_rng_state", None) is not None:
                torch.set_rng_state(st["torch_rng_state"])
            if torch.cuda.is_available() and st.get("torch_cuda_rng_state_all", None) is not None:
                torch.cuda.set_rng_state_all(st["torch_cuda_rng_state_all"])
        else:
            # optional JSON resume: use saved solution as new x0 (+ optional sigma)
            if resume_json and os.path.isfile(resume_json):
                with open(resume_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                x0 = np.asarray(payload["solution"], dtype=np.float64)
                self.generation = int(payload.get("step", 0))
                self.best_solution = x0.copy()
                self.best_train = float(payload.get("train_best", -np.inf))
                v = payload.get("val", None)
                self.best_val = float(v) if v is not None else float("-inf")
                if "sigma" in payload:
                    opts["sigma0"] = float(payload["sigma"])
            else:
                self._train_iter = iter(self.train_loader) if self.train_loader is not None else None
                self.best_solution = x0.copy()
                self.best_train = -np.inf
                self.best_val = -np.inf

            self.es = cma.CMAEvolutionStrategy(
                x0=x0,
                sigma0=float(cfg.optimize.initial_sigma),
                inopts=opts,
            )

        # history
        self.hist_train_best: List[float] = []
        self.hist_train_best_scores: List[float] = []
        self.hist_train_mean: List[float] = []
        self.hist_val: List[Optional[float]] = []
        self.hist_sigma: List[float] = []
        self.hist_models_scales: List[List[float]] = []

    def _is_dist(self):
        return dist.is_available() and dist.is_initialized()

    def _rank(self):
        return dist.get_rank() if self._is_dist() else 0

    def _world(self):
        return dist.get_world_size() if self._is_dist() else 1

    def _barrier(self):
        if self._is_dist():
            dist.barrier()

    def _broadcast_object(self, obj, src=0):
        if not self._is_dist():
            return obj
        container = [obj]
        dist.broadcast_object_list(container, src=src)
        return container[0]

    def _shard_list(self, xs):
        if self._world() <= 1:
            return list(xs)
        r, w = self._rank(), self._world()
        return list(xs)[r::w]

    def _allreduce_count(self, n_local: int) -> int:
        if not self._is_dist():
            return n_local
        t = torch.tensor([int(n_local)], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return int(t.item())

    def _allreduce_score_sum(self, d_local: dict) -> dict:
        # Стабильный порядок ключей
        keys = sorted(d_local.keys())
        vals = torch.tensor([float(d_local[k]) for k in keys], device=self.device)
        if self._is_dist():
            dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        return {k: float(v) for k, v in zip(keys, vals.tolist())}

    def _next_batch(self) -> List[str]:
        batch = None
        if self._rank() == 0:
            try:
                batch = next(self._train_iter)
            except StopIteration:
                self._train_iter = iter(self.train_loader)
                batch = next(self._train_iter)
            batch = list(batch)
        # один и тот же batch на всех рангах
        batch = self._broadcast_object(batch, src=0)
        return list(batch)

    def _apply_x_via_pipeline(self, x: np.ndarray):
        scales_double, scales_single, models_scales = _unflatten_to_gs(x, self.shapes)
        self.pipeline.modify_scaleguidance(
            num_models=self.shapes["n_models"],
            scales_double=scales_double,
            scales_single=scales_single,
            models_scales=models_scales,
        )

    def _gen_images_batch(self, prompts: List[str], seed: int):
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

    # def _eval_candidate_on_bucket(self, x: np.ndarray, bucket_prompts: List[str], seed: int) -> Dict[str, float]:
    #     self._apply_x_via_pipeline(x)
    #     total_scores = None
    #     total_count = 0
    #     micro_bs = int(self.cfg.data.batch_size)

    #     for mini in _iter_chunks(bucket_prompts, micro_bs):
    #         mini = list(mini)
    #         shard_prompts = self._shard_list(mini)  # NEW

    #         images = self._gen_images_batch(shard_prompts, seed)
    #         scores = call_reward(self.reward_fn, images, shard_prompts)
    #         sum_scores_local = mean_score(scores, mode="sum")

    #         # Глобальные суммы по всем рангам
    #         sum_scores = self._allreduce_score_sum(sum_scores_local)  # NEW
    #         count = self._allreduce_count(len(shard_prompts))         # NEW

    #         if total_scores is None:
    #             total_scores = sum_scores
    #         else:
    #             for name, s in sum_scores.items():
    #                 total_scores[name] += s
    #         total_count += count

    #     return {name: float(score) / float(total_count) for name, score in total_scores.items()}

    def _eval_candidate_on_bucket(self, x: np.ndarray, bucket_prompts: List[str], seed: int) -> Dict[str, float]:
        self._apply_x_via_pipeline(x)
        total_scores = None
        total_count = 0
        micro_bs = int(self.cfg.data.batch_size)
        for mini in _iter_chunks(bucket_prompts, micro_bs):
            mini = list(mini)  # ВАЖНО: не шардируем по рангам
            images = self._gen_images_batch(mini, seed)
            scores = call_reward(self.reward_fn, images, mini)
            sum_scores = mean_score(scores, mode="sum")
            if total_scores is None:
                total_scores = sum_scores
            else:
                for name, s in sum_scores.items():
                    total_scores[name] += s
            total_count += len(mini)
        return {name: float(score) / float(total_count) for name, score in total_scores.items()}

    def _eval_validation(self, x: np.ndarray, seed: int = 1234,
                         save_dir: str | None = None, save_images: bool = False,
                         save_format: str = "png", save_limit: int | None = None) -> dict | None:
        self._apply_x_via_pipeline(x)
        total_scores = None
        total_count = 0
        global_idx = 0
        gen_device = "cpu"

        for prompts in tqdm(self.val_loader, desc="Validating", total=len(self.val_loader)):
            prompts = list(prompts)
            shard_prompts = self._shard_list(prompts)

            generator = torch.Generator(device=gen_device).manual_seed(int(seed))
            with torch.inference_mode():
                out = self.pipeline(
                    list(shard_prompts),
                    num_inference_steps=self.gen_params["num_inference_steps"],
                    guidance_scale=self.gen_params["guidance_scale"],
                    height=self.gen_params["height"],
                    width=self.gen_params["width"],
                    generator=generator,
                )
            images = out.images

            if save_images and save_dir is not None and self._rank() == 0:
                os.makedirs(save_dir, exist_ok=True)
                pil_images = to_pil_list(images)
                for im in pil_images:
                    if save_limit is not None and global_idx >= save_limit:
                        break
                    fname = f"{global_idx:06d}.{save_format}"
                    im.save(os.path.join(save_dir, fname), format=save_format.upper())
                    global_idx += 1

            if self.eval_reward_fn:
                scores = call_reward(self.eval_reward_fn, images, list(shard_prompts))
                sum_scores_local = mean_score(scores, mode="sum")
                sum_scores = self._allreduce_score_sum(sum_scores_local)
                count = self._allreduce_count(len(shard_prompts))

                if total_scores is None:
                    total_scores = sum_scores
                else:
                    for name, score in sum_scores.items():
                        total_scores[name] += score
                total_count += count

        if self.eval_reward_fn:
            return {name: score / total_count for name, score in total_scores.items()}
        else:
            return None


    def _log_step(self, step: int, best_fit: float, mean_fit: float, sigma: float, best_x: np.ndarray):
        scalars = {
            "train/best": float(best_fit),
            "train/mean": float(mean_fit),
            "cmaes/sigma": float(sigma),
        }
        log_scalars(self.writer, scalars, step)

        # Восстанавливаем по-модельные коэффициенты
        scales_double, scales_single, models_scales = _unflatten_to_gs(best_x, self.shapes)
        n_models = self.shapes["n_models"]

        # Разворачиваем в per-model массивы: attn, mlp, single
        attn_by_model = [np.array([a for (a, _m) in scales_double[m]], dtype=np.float32) for m in range(n_models)]
        mlp_by_model  = [np.array([m_ for (_a, m_) in scales_double[m]], dtype=np.float32) for m in range(n_models)]
        single_by_model = [np.array(scales_single[m], dtype=np.float32) for m in range(n_models)]

        alpha_dict = {
            "double_attn": attn_by_model,         # список np.array по моделям
            "double_mlp": mlp_by_model,           # список np.array по моделям
            "single": single_by_model,            # список np.array по моделям
            "models_scales": np.array(models_scales, dtype=np.float32),  # shape = [n_models]
        }
        log_scatter(self.writer, alpha_dict, step, prefix="alphas/")

        self.hist_sigma.append(float(sigma))
        self.hist_models_scales.append(np.asarray(models_scales, dtype=np.float32).tolist())

    def _maybe_log_images(self, step: int, x: np.ndarray, tag: str = "samples/test"):
        if not getattr(self.cfg.experiment, "test_dataset", False):
            return
        prompts = get_lines(self.cfg.experiment.test_dataset)
        self._apply_x_via_pipeline(x)
        show_seed = int(getattr(self.cfg.experiment, "seed", 0)) + step + 777
        images = self._gen_images_batch(prompts, show_seed)
        log_images(self.writer, tag, images, step)

    def _checkpoint_json(self, step: int, best_fit: float, mean_fit: float, val_fit: Optional[float], best_x: np.ndarray):
        if getattr(self.cfg.experiment, "save_json", None) is None:
            return
        if self._rank() != 0:
            return
        # run_dir = getattr(self.writer, "log_dir", None)
        run_dir = self.logdir
        run_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(run_dir, exist_ok=True)
        payload = {
            "step": int(step),
            "train_best": float(best_fit),
            "train_mean": float(mean_fit),
            "val": None if val_fit is None else float(val_fit),
            "solution": [float(v) for v in list(best_x)],
            "shapes": self.shapes,
            "sigma": float(self.es.sigma),
            "timestamp": int(time.time()),
        }
        with open(os.path.join(run_dir, f"cmaes_step_{step:04d}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _state_path(self, step: int) -> str:
        run_dir = os.path.join(self.logdir, "checkpoints", f"step_{step}")
        os.makedirs(run_dir, exist_ok=True)
        return os.path.join(run_dir, f"cmaes_state_{step:04d}.pkl")

    def _save_state(self, step: int):
        if self._rank() != 0:
            return
        st = {
            "generation": int(step),
            "es": self.es,
            "best_solution": self.best_solution,
            "best_train": float(self.best_train),
            "best_val": float(self.best_val),
            "hist_train_best": self.hist_train_best,
            "hist_train_mean": self.hist_train_mean,
            "hist_train_best_scores": self.hist_train_best_scores,
            "hist_val": self.hist_val,
            "hist_sigma": self.hist_sigma,
            "hist_models_scales": self.hist_models_scales,
            "np_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        with open(self._state_path(step), "wb") as f:
            pickle.dump(st, f)


    def train(self) -> Tuple[np.ndarray, float, float]:
        generation = 0
        max_gens = int(self.cfg.optimize.max_generations)
        base_seed = int(getattr(self.cfg.experiment, "seed", 0))

        ### eval orig model
        if self.cfg.experiment.eval_orig_model:
            orig_x = _flatten_from_transformer(self.pipeline)
            val_score = self._eval_validation(orig_x, 
                                              save_dir=os.path.join(self.logdir, "eval", f"step_{generation}"), 
                                              save_images=getattr(self.cfg.data, "save_eval_imgs", None),
                                              seed=1234)
            if self._rank() == 0:
                for name, score in val_score.items():
                    log_scalars(self.writer, {f"val/{name}": float(score)}, generation)
                self._maybe_log_images(generation, orig_x)
                self._checkpoint_json(generation, -1.0, -1.0, val_score["avg"], orig_x)
                self._save_state(generation)

        while not self.es.stop() and (generation < max_gens or max_gens < 0):
            step = generation + 1

            # one shared train batch per generation
            batch_prompts = self._next_batch()

            # evaluate candidates on the same batch
            if self._rank() == 0:
                candidates = self.es.ask()
                log_scalars(self.writer, {f"train/num_candidates": len(candidates)}, step)
            else:
                candidates = None
            candidates = self._broadcast_object(candidates, src=0)  # одинаковый список у всех

            # shards by candidates
            idxs = list(range(len(candidates)))
            my_idxs = idxs[self._rank()::self._world()]
            my_cands = [candidates[i] for i in my_idxs]

            my_fit, my_scr = [], []
            for i, x in tqdm(zip(my_idxs, my_cands), desc="Iter over candidates", total=len(my_cands)):
                sc = self._eval_candidate_on_bucket(x, batch_prompts, seed=base_seed + generation)
                my_fit.append((i, -float(sc["avg"])))
                my_scr.append((i, sc))

            # сбор результатов на rank 0 и tell()
            if self._is_dist():
                recv_fit = [None] * self._world() if self._rank() == 0 else None
                recv_scr = [None] * self._world() if self._rank() == 0 else None
                dist.gather_object(my_fit, recv_fit, dst=0)
                dist.gather_object(my_scr, recv_scr, dst=0)
            else:
                recv_fit, recv_scr = [my_fit], [my_scr]

            if self._rank() == 0:
                pairs_fit = [p for part in recv_fit for p in part]
                pairs_scr = [p for part in recv_scr for p in part]
                pairs_fit.sort(key=lambda t: t[0])
                pairs_scr.sort(key=lambda t: t[0])
                fitnesses = [f for _, f in pairs_fit]
                scores = [s for _, s in pairs_scr]
                self.es.tell(candidates, fitnesses)

                best_idx = int(np.argmin(fitnesses))
                best_x = np.asarray(candidates[best_idx], dtype=np.float64)
                best_fit = -float(fitnesses[best_idx])
                best_scores = scores[best_idx]
                mean_fit = -float(np.mean(fitnesses))

                if best_fit > self.best_train:
                    self.best_train = best_fit
                    self.best_train_scores = best_scores
                    self.best_solution = best_x.copy()

                self.hist_train_best.append(best_fit)
                self.hist_train_mean.append(mean_fit)
                self.hist_train_best_scores.append(best_scores)

                self._log_step(step, best_fit, mean_fit, float(self.es.sigma), best_x)

            self._barrier()

            do_val = int(self.cfg.optimize.val_every_steps) > 0 and (step % int(self.cfg.optimize.val_every_steps) == 0 or step == max_gens)
            if do_val:
                best_sol = self._broadcast_object(self.best_solution if self._rank() == 0 else None, src=0)
                best_sol = np.asarray(best_sol, dtype=np.float64)
                val_score = self._eval_validation(
                    best_sol,
                    save_dir=os.path.join(self.logdir, "eval", f"step_{step}"),
                    save_images=getattr(self.cfg.data, "save_eval_imgs", None),
                    seed=1234
                )
                if self._rank() == 0:
                    self.hist_val.append(val_score)
                    for name, score in val_score.items():
                        log_scalars(self.writer, {f"val/{name}": float(score)}, step)
                    self._maybe_log_images(step, best_sol)
                    if val_score["avg"] > self.best_val:
                        self.best_val = float(val_score["avg"])
                    self._checkpoint_json(step, self.best_train, float(self.hist_train_mean[-1]), self.hist_val[-1]["avg"], best_sol)
                    self._save_state(step)
            else:
                if self._rank() == 0:
                    self.hist_val.append(None)

            generation += 1

        if self._rank() == 0:
            last_mean = float(self.hist_train_mean[-1]) if self.hist_train_mean else float("-inf")
            last_val = self.best_val if (self.hist_val and self.hist_val[-1] is not None) else None
            self._checkpoint_json(generation, self.best_train, last_mean, last_val, self.best_solution)
            self._save_state(generation)

        result = self._broadcast_object(
            dict(x=self.best_solution.tolist() if self._rank() == 0 else None,
                 best_train=float(self.best_train if self._rank() == 0 else 0.0),
                 best_val=float(self.best_val if self._rank() == 0 else 0.0)), 
            src=0
        )
        best_solution = np.asarray(result["x"], dtype=np.float64)
        return best_solution, result["best_train"], result["best_val"]
