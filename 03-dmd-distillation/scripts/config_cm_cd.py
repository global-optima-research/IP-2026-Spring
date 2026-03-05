# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import fastgen.configs.methods.config_cm as config_cm_default
from fastgen.configs.data import VideoLoaderConfig
from fastgen.configs.net import Wan_1_3B_Config
from fastgen.utils import LazyCall as L
from fastgen.utils.lr_scheduler import LambdaInverseSquareRootScheduler
from fastgen.callbacks.ema import EMACallback

"""
Configs for Consistency Distillation (CD) on Wan2.1-1.3B.

CD = CM with use_cd=True (requires teacher for ODE solving).
Adapted from EDM2/config_cm_s.py + WanT2V/config_dmd2.py.

References:
  - Song et al., 2023: Consistency Models (https://arxiv.org/abs/2303.01469)
  - Geng et al., 2024: ECT (https://arxiv.org/abs/2406.14548)
"""


def create_config():
    config = config_cm_default.create_config()

    # === CT Schedule (curriculum for t-r distance) ===
    config.trainer.callbacks.ct_schedule.kimg_per_stage = 50
    config.trainer.callbacks.ct_schedule.q = 4
    config.trainer.callbacks.ct_schedule.ratio_limit = 0.9961

    # === EMA ===
    config.model.use_ema = ["ema_1"]
    from omegaconf import DictConfig
    config.trainer.callbacks = DictConfig(
        {k: v for k, v in config.trainer.callbacks.items() if not k.startswith("ema")}
    )
    config.trainer.callbacks.update(
        dict(ema_1=L(EMACallback)(type="power", gamma=96.99, ema_name="ema_1"))
    )

    # === Model ===
    config.model.precision = "bfloat16"
    config.model.input_shape = [16, 21, 60, 104]

    # CD: use teacher for ODE-based target (use_cd=True)
    config.model.loss_config.use_cd = True
    config.model.loss_config.huber_const = 0.06
    config.model.loss_config.weighting_ct_loss = "default"

    # CFG guidance for teacher (same as DMD2)
    config.model.guidance_scale = 5.0

    # Network: Wan2.1-1.3B
    config.model.net = copy.deepcopy(Wan_1_3B_Config)
    config.model.enable_preprocessors = False

    # Optimizer
    config.model.net_optimizer.optim_type = "adamw"
    config.model.net_optimizer.lr = 1e-5
    config.model.net_optimizer.betas = (0.9, 0.99)
    config.model.net_optimizer.eps = 1e-8
    config.model.net_optimizer.weight_decay = 0.0
    config.model.net_scheduler = L(LambdaInverseSquareRootScheduler)(
        warm_up_steps=0,
        decay_steps=2000,
    )

    # Time sampling
    config.model.sample_t_cfg.time_dist_type = "logitnormal"
    config.model.sample_t_cfg.train_p_mean = -0.8
    config.model.sample_t_cfg.train_p_std = 1.6
    config.model.sample_t_cfg.min_t = 0.001
    config.model.sample_t_cfg.max_t = 0.999

    # === Dataloader: same format as DMD2 (mp4 + txt WebDataset) ===
    config.dataloader_train = VideoLoaderConfig
    config.dataloader_train.batch_size = 1

    # 480p (832x480) resolution
    config.dataloader_train.img_size = (
        config.model.input_shape[-1] * 8,
        config.model.input_shape[-2] * 8,
    )
    config.dataloader_train.sequence_length = (config.model.input_shape[1] - 1) * 4 + 1

    # === Training ===
    config.trainer.max_iter = 6000
    config.trainer.logging_iter = 50
    config.trainer.save_ckpt_iter = 500
    config.trainer.batch_size_global = 8

    config.log_config.group = "wan_cm_cd"
    return config
