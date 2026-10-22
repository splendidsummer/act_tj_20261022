"""Diffusion policy with the same infer(obs) -> dict interface as the ACT server."""

from pathlib import Path

import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.control_utils import predict_action

from serving.act_policy import ACTION_NAMES
from serving.act_policy import _has_images
from serving.act_policy import observation_from_client
from serving.quad_image import CAMERAS as SHORT_CAMERAS
from serving.quad_image import QuadImageDecoder


class DiffusionServePolicy:
    def __init__(self, ckpt: str | Path, device: str = "cuda", chunk_size: int | None = None) -> None:
        self.ckpt = Path(ckpt)
        self.device = torch.device(device)
        self.policy = DiffusionPolicy.from_pretrained(self.ckpt)
        self.policy.to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.ckpt,
        )
        executable = int(self.policy.config.n_action_steps)
        if chunk_size is None or chunk_size > executable:
            self.return_chunk = executable
        elif chunk_size < 1:
            raise ValueError(f"--chunk-size must be at least 1, got {chunk_size}")
        else:
            self.return_chunk = chunk_size
        self._quad_decoder = QuadImageDecoder()
        self.reset()

    @property
    def metadata(self) -> dict:
        return {
            "policy": "diffusion",
            "server": "serve_policy_diffusion",
            "ckpt": str(self.ckpt),
            "checkpoint_dir": str(self.ckpt),
            "state_dim": 21,
            "action_dim": 21,
            "image_shape": [744, 960, 3],
            "image_height": 744,
            "image_width": 960,
            "cameras": list(SHORT_CAMERAS),
            "fps": 30,
            "action_rate": 30.0,
            "action_mode": "absolute",
            "output_action_schema": list(ACTION_NAMES),
            "n_obs_steps": self.policy.config.n_obs_steps,
            "horizon": self.policy.config.horizon,
            "n_action_steps": self.policy.config.n_action_steps,
            "chunk_size": self.return_chunk,
        }

    def reset(self) -> None:
        self.policy.reset()

    def infer(self, obs: dict) -> dict:
        if obs.get("reset"):
            self.reset()
            if not _has_images(obs):
                return {"reset": True}

        observation, decode_ms = observation_from_client(self._quad_decoder, obs)
        # A new client request replans. Keep the observation history, drop leftover actions.
        self.policy._queues["action"].clear()
        steps = []
        for _ in range(self.return_chunk):
            action = predict_action(
                observation,
                self.policy,
                self.device,
                self.preprocessor,
                self.postprocessor,
                use_amp=False,
                task="remove the object",
                robot_type="gento_smoke",
            )
            steps.append(action.squeeze(0).detach().cpu().numpy().astype(np.float32))
        result = {"actions": np.stack(steps, axis=0)}
        if decode_ms is not None:
            result["image_decode_ms"] = decode_ms
        return result
