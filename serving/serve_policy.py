"""Serve an ACT or Diffusion checkpoint over the OpenPI websocket protocol.

This is the Gento policy server for LeRobot checkpoints. It is not
scripts/serve_policy_gento_ws.py in openpi_tj_data_collector. That process
loads Pi05 and resizes each camera to 224. This process keeps each tile at
the training size, 960x744, then blacks out the bottom 48 rows of the two
wrist images. The client must send the undistorted quad. Actions are
absolute 21-D joint targets.

    conda activate /home/holdon/Projects/act_tj/.venv
    cd /home/holdon/Projects/act_tj
    python -m serving.serve_policy --ckpt <pretrained_model> --port 8000 --chunk-size 30

The policy class is read from the checkpoint config.json. Pass --chunk-size 30
for both policies. ACT returns 30 actions. Diffusion clamps the return to
n_action_steps (8 for the MoveParcel run) and the client must execute only
that many. Pass 0 with an ACT checkpoint to return one action per request.
"""

import argparse
import json
import logging
import socket
from pathlib import Path

from serving.websocket_policy_server import WebsocketPolicyServer

DEFAULT_CKPT = (
    "/home/holdon/Projects/act_tj/outputs/train/act_move_parcel/checkpoints/100000/pretrained_model"
)


def load_policy(ckpt: str, device: str, chunk_size: int | None):
    config_path = Path(ckpt) / "config.json"
    policy_type = json.loads(config_path.read_text())["type"]
    if policy_type == "diffusion":
        from serving.diffusion_policy import DiffusionServePolicy

        return DiffusionServePolicy(ckpt, device=device, chunk_size=chunk_size)
    if policy_type == "act":
        from serving.act_policy import ActPolicy

        return ActPolicy(ckpt, device=device, chunk_size=chunk_size)
    raise ValueError(f"Unsupported policy type {policy_type!r} in {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Actions returned to the Gento client. 30 matches --replan-steps 30. 0 returns one action.",
    )
    args = parser.parse_args()
    chunk_size = None if args.chunk_size <= 0 else args.chunk_size

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    policy = load_policy(args.ckpt, args.device, chunk_size)
    logging.info("Loaded %s (%s)", args.ckpt, policy.metadata["policy"])
    if policy.metadata["policy"] == "diffusion" and chunk_size != policy.metadata["chunk_size"]:
        logging.info(
            "Diffusion returns %s actions (n_action_steps); client --replan-steps must be <= this",
            policy.metadata["chunk_size"],
        )
    logging.info("Host %s ip %s", socket.gethostname(), socket.gethostbyname(socket.gethostname()))

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
