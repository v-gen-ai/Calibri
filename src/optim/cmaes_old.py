# trainers/cmaes_trainer.py
import os
import json
import time
from typing import Dict, List, Optional

import numpy as np
import torch

import cma
import torchvision


def _pil_to_tensor(img):
    # (H, W, C) uint8 -> (C, H, W) float in [0,1]
    import torchvision.transforms.functional as F
    return F.to_tensor(img)


class CMAESTrainer:
    """
    Полная перепись тренера CMA-ES с батчевой оценкой:
    - На каждое поколение эволюции берётся ровно один следующий батч из train_loader.
    - Все кандидаты поколения оцениваются на этом одном и том же батче (общая стохастика).
    - Валидация выполняется по расписанию и считается по всей val-выборке батчами.
    - Логи и примеры изображений пишутся в TensorBoard; чекпоинты сохраняются по расписанию.
    """

    def __init__(self, cfg, pipeline, reward_fn, train_loader, val_loader):
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(cfg.model.device)

        # Ревард-модель
        self.reward_fn = reward_fn

        # Размерность вектора решения: [main_alphas | guide_alphas | guidance_weight]
        n_double, n_single, dim_per_model, _ = self.pipeline.dims()
        self.dim_per_model = dim_per_model
        self.total_dim = dim_per_model * 2 + 1

        # Инициализация стартовой точки
        x0 = np.concatenate([
            np.full(dim_per_model, 1.0, dtype=np.float64),   # main alphas
            np.full(dim_per_model, 1.0, dtype=np.float64),   # guidance alphas
            np.array([cfg.autog.initial_guidance_weight], dtype=np.float64),
        ])

        # Границы: alphas по alpha_bounds, вес guidance [0,1]
        bounds_low = [cfg.autog.alpha_bounds[0]] * (dim_per_model * 2) + [0.0]
        bounds_high = [cfg.autog.alpha_bounds[1]] * (dim_per_model * 2) + [1.0]

        # Настройки CMA-ES
        opts = {
            "bounds": [bounds_low, bounds_high],
            "verb_disp": cfg.cmaes.verb_disp,
            "verb_log": cfg.cmaes.verb_log,
            "maxiter": cfg.cmaes.max_generations,
            "seed": cfg.cmaes.seed,
        }
        if cfg.cmaes.population_size:
            opts["popsize"] = cfg.cmaes.population_size

        self.es = cma.CMAEvolutionStrategy(x0, cfg.cmaes.initial_sigma, opts)

        # TensorBoard
        run_dir = f"{cfg.log_dir}/{cfg.experiment_name}_{int(time.time())}"
        os.makedirs(run_dir, exist_ok=True)
        self.writer = SummaryWriter(run_dir)
        self.run_dir = run_dir

        # Истории метрик
        self.best_fitness_history: List[float] = []
        self.mean_fitness_history: List[float] = []
        self.validation_fitness_history: List[Optional[float]] = []
        self.guidance_weight_history: List[float] = []
        self.sigma_history: List[float] = []

        # Ранняя остановка
        self.best_val = -np.inf
        self.patience = 0

        # Stateful итератор по train
        self.train_iter = iter(self.train_loader)

    def _next_train_batch(self) -> List[str]:
        try:
            return next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            return next(self.train_iter)

    def _apply_params(self, x: np.ndarray) -> float:
        """
        Применяет вектор параметров к пайплайну и возвращает guidance_weight.
        """
        _, _, gw = self.pipeline.vector_to_params(x, self.cfg.autog.alpha_bounds)
        return float(gw)

    def _eval_on_batch(
        self,
        x: np.ndarray,
        batch_prompts: List[str],
        gen_params: Dict,
        seed: int,
    ) -> float:
        """
        Оценка среднего reward на заданном списке промптов (один батч).
        """
        gw = self._apply_params(x)
        generator = torch.Generator(device=self.device if self.device.type == "cuda" else "cpu").manual_seed(int(seed))
        images = self.pipeline.generate_batch(
            prompts=batch_prompts,
            guidance_weight=gw,
            num_inference_steps=gen_params["num_inference_steps"],
            guidance_scale=gen_params["guidance_scale"],
            height=gen_params["image_size"],
            width=gen_params["image_size"],
            generator=generator,
        )
        scores = self.reward.score_batch(batch_prompts, images)
        return float(np.mean(scores)) if len(scores) else -np.inf

    def _eval_validation(
        self,
        x: np.ndarray,
        gen_params: Dict,
        seed: int = 1234,
    ) -> float:
        """
        Агрегированная валидация по всей val-выборке (батчами).
        """
        gw = self._apply_params(x)
        total, count = 0.0, 0
        for batch_prompts in self.val_loader:
            generator = torch.Generator(device=self.device if self.device.type == "cuda" else "cpu").manual_seed(int(seed))
            images = self.pipeline.generate_batch(
                prompts=batch_prompts,
                guidance_weight=gw,
                num_inference_steps=gen_params["num_inference_steps"],
                guidance_scale=gen_params["guidance_scale"],
                height=gen_params["image_size"],
                width=gen_params["image_size"],
                generator=generator,
            )
            scores = self.reward.score_batch(batch_prompts, images)
            total += float(np.sum(scores))
            count += len(scores)
        return float(total / max(1, count))

    def _log_images(self, tag: str, prompts: List[str], images: List[torch.Tensor], step: int):
        n = min(len(images), self.cfg.max_images_to_log)
        if n <= 0:
            return
        grid = torchvision.utils.make_grid(
            torch.stack([_pil_to_tensor(im) for im in images[:n]]),
            nrow=n,
            padding=2,
        )
        self.writer.add_image(tag, grid, global_step=step)

    def _log_alphas(self, x: np.ndarray, step: int):
        n = self.dim_per_model
        main = x[:n]
        guide = x[n: 2 * n]
        self.writer.add_histogram("alphas/main", torch.tensor(main), step)
        self.writer.add_histogram("alphas/guidance", torch.tensor(guide), step)
        self.writer.add_scalar("guidance/weight", float(x[2 * n]), step)

    def _checkpoint(self, generation: int, best_fit: float, mean_fit: float, val_fit: Optional[float], gw: float):
        if not self.cfg.save_checkpoints:
            return
        if not (generation % self.cfg.save_every == 0 or generation == 0):
            return
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        ckpt = {
            "generation": generation,
            "best_fitness": float(best_fit),
            "mean_fitness": float(mean_fit),
            "val_fitness": None if val_fit is None else float(val_fit),
            "guidance_weight": float(gw),
        }
        path = os.path.join(self.cfg.checkpoint_dir, f"ckpt_gen{generation:03d}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2, ensure_ascii=False)

    def train(self) -> Dict[str, List]:
        gen_params = {
            "num_inference_steps": self.cfg.gen.num_inference_steps,
            "guidance_scale": self.cfg.gen.guidance_scale,
            "image_size": self.cfg.gen.image_size,
        }

        generation = 0
        max_gens = int(self.cfg.cmaes.max_generations)

        while not self.es.stop() and generation < max_gens:
            step = generation + 1

            # 1) Один следующий train-батч для всего поколения
            batch_prompts = self._next_train_batch()

            # 2) Оценка всех кандидатов на этом батче
            solutions = self.es.ask()
            base_seed = int(self.cfg.seed) + generation  # одинаковая стохастика внутри поколения
            fitnesses = []
            for x in solutions:
                # CMA-ES минимизирует => берем отрицание reward
                f = -self._eval_on_batch(
                    x=x,
                    batch_prompts=batch_prompts,
                    gen_params=gen_params,
                    seed=base_seed,
                )
                fitnesses.append(float(f))

            # 3) Обновление стратегии
            self.es.tell(solutions, fitnesses)

            # 4) Подсчёт метрик поколения
            best_idx = int(np.argmin(fitnesses))
            best_x = solutions[best_idx]
            best_fit = -float(fitnesses[best_idx])
            mean_fit = -float(np.mean(fitnesses))
            gw = self._apply_params(best_x)

            self.best_fitness_history.append(best_fit)
            self.mean_fitness_history.append(mean_fit)
            self.sigma_history.append(float(self.es.sigma))
            self.guidance_weight_history.append(float(gw))

            self.writer.add_scalar("train/best", best_fit, step)
            self.writer.add_scalar("train/mean", mean_fit, step)
            self.writer.add_scalar("cmaes/sigma", float(self.es.sigma), step)
            self._log_alphas(best_x, step)

            # 5) Лог изображений на том же батче
            if self.cfg.log_images_every and (step % self.cfg.log_images_every == 0 or step == 1):
                images = self.pipeline.generate_batch(
                    prompts=batch_prompts,
                    guidance_weight=gw,
                    num_inference_steps=gen_params["num_inference_steps"],
                    guidance_scale=gen_params["guidance_scale"],
                    height=gen_params["image_size"],
                    width=gen_params["image_size"],
                    generator=torch.Generator(device=self.device if self.device.type == "cuda" else "cpu").manual_seed(base_seed + 777),
                )
                self._log_images("samples/train", batch_prompts, images, step)

            # 6) Валидация по расписанию
            do_val = self.cfg.train.val_every and (step % self.cfg.train.val_every == 0 or step == max_gens)
            if do_val:
                val_score = self._eval_validation(best_x, gen_params, seed=1234)
                self.validation_fitness_history.append(val_score)
                self.writer.add_scalar("val/mean", float(val_score), step)

                # Ранняя остановка/переобучение
                train_val_gap = best_fit - float(val_score)
                if val_score > self.best_val:
                    self.best_val = float(val_score)
                    self.patience = 0
                else:
                    self.patience += 1

                should_stop = False
                if self.patience >= int(self.cfg.train.early_stopping_patience):
                    should_stop = True
                if train_val_gap > float(self.cfg.train.overfitting_threshold) and self.patience >= int(self.cfg.train.early_stopping_patience):
                    should_stop = True

                if should_stop:
                    self._checkpoint(generation, best_fit, mean_fit, val_score, gw)
                    break
            else:
                self.validation_fitness_history.append(None)

            # 7) Чекпоинт
            self._checkpoint(generation, best_fit, mean_fit, self.validation_fitness_history[-1], gw)

            generation += 1

        self.writer.flush()
        self.writer.close()

        return {
            "train_best": self.best_fitness_history,
            "train_mean": self.mean_fitness_history,
            "val": self.validation_fitness_history,
            "sigma": self.sigma_history,
            "guidance_weight": self.guidance_weight_history,
            "run_dir": self.run_dir,
        }
