"""ACT policy with the same infer(obs) -> dict interface as OpenPI."""

from pathlib import Path

import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.control_utils import predict_action

from serving.quad_image import CAMERAS as SHORT_CAMERAS
from serving.quad_image import QuadImageDecoder

CAMERAS = tuple(f"observation.images.{name}" for name in SHORT_CAMERAS)
ACTION_NAMES = (
    *(f"left_joint_{index}" for index in range(7)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(7)),
    "right_gripper",
    "lift",
    "waist_yaw",
    "waist_pitch",
    "head_yaw",
    "head_pitch",
)


class ActPolicy:
    def __init__(self, ckpt: str | Path, device: str = "cuda", chunk_size: int | None = None) -> None:
        self.ckpt = Path(ckpt)
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(self.ckpt)
        self.policy.to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.ckpt,
        )
        model_chunk = int(self.policy.config.chunk_size)
        if chunk_size is None:
            self.return_chunk = None
        else:
            if not 1 <= chunk_size <= model_chunk:
                raise ValueError(f"--chunk-size must be in 1..{model_chunk}, got {chunk_size}")
            self.return_chunk = chunk_size
        self._quad_decoder = QuadImageDecoder()
        self.reset()

    @property
    def metadata(self) -> dict:
        returned = self.return_chunk if self.return_chunk is not None else self.policy.config.chunk_size
        return {
            "policy": "act",
            "server": "serve_policy_act",
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
            "temporal_ensemble_coeff": self.policy.config.temporal_ensemble_coeff,
            "model_chunk_size": self.policy.config.chunk_size,
            "chunk_size": returned,
        }

    def reset(self) -> None:
        self.policy.reset()

    def infer(self, obs: dict) -> dict:
        if obs.get("reset"):
            self.reset()
            if not _has_images(obs):
                return {"reset": True}

        observation, decode_ms = self._observation(obs)
        if self.return_chunk is None:
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
            actions = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        else:
            actions = self._chunk(observation)
        result = {"actions": actions}
        if decode_ms is not None:
            result["image_decode_ms"] = decode_ms
        return result

    def _observation(self, obs: dict) -> tuple[dict, float | None]:
        return observation_from_client(self._quad_decoder, obs)

    def _chunk(self, observation: dict) -> np.ndarray:
        with torch.inference_mode():
            batch = prepare_observation_for_inference(
                observation,
                self.device,
                task="remove the object",
                robot_type="gento_smoke",
            )
            batch = self.preprocessor(batch)
            actions = self.policy.predict_action_chunk(batch)
            actions = self.postprocessor(actions)
        actions = actions.detach().cpu().numpy().astype(np.float32)
        if actions.ndim == 3:
            actions = actions[0]
        return np.ascontiguousarray(actions[: self.return_chunk])


def observation_from_client(decoder, obs: dict) -> tuple[dict, float | None]:
    decode_ms = None
    if "compressed_image" in obs:
        tiles, decode_ms = decoder.decode(obs["compressed_image"])
        images = {f"observation.images.{name}": tiles[name] for name in SHORT_CAMERAS}
        state = obs.get("state", obs.get("observation.state"))
    elif isinstance(obs.get("images"), dict):
        images = {f"observation.images.{name}": obs["images"][name] for name in SHORT_CAMERAS}
        state = obs.get("state", obs.get("observation.state"))
    else:
        images = {key: obs[key] for key in CAMERAS}
        state = obs["observation.state"]
    if state is None:
        raise ValueError("observation must contain 'state' or 'observation.state'")
    observation = {"observation.state": _state(state)}
    for key, value in images.items():
        observation[key] = _image(value)
    return observation, decode_ms


def _has_images(obs: dict) -> bool:
    if "compressed_image" in obs or isinstance(obs.get("images"), dict):
        return True
    return any(key in obs for key in CAMERAS)


def _state(value) -> np.ndarray:
    state = np.asarray(value, dtype=np.float32)
    return state.reshape(21)


def _image(value) -> np.ndarray:
    image = np.asarray(value)
    if image.shape[0] == 3 and image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).round().astype(np.uint8)
    if image.shape != (744, 960, 3):
        raise ValueError(f"Expected image shape (744, 960, 3) or (3, 744, 960), got {image.shape}")
    return image
