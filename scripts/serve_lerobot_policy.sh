#!/usr/bin/env bash
# LeRobot policy server for the Gento client.
# Run this on the GPU machine, from any cwd. Do not start openpi's
# scripts/serve_policy_gento_ws.py for these checkpoints.
#
# ACT, return 30 actions:
#   CKPT=/path/to/act/pretrained_model bash scripts/serve_lerobot_policy.sh
#
# Diffusion, --chunk-size 30 is clamped to n_action_steps (8):
#   CKPT=/path/to/diffusion/pretrained_model bash scripts/serve_lerobot_policy.sh
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${CKPT:?set CKPT to a pretrained_model directory}"
PORT="${PORT:-8000}"
CHUNK_SIZE="${CHUNK_SIZE:-30}"
HOST="${HOST:-0.0.0.0}"

exec .venv/bin/python -m serving.serve_policy \
  --ckpt "$CKPT" \
  --host "$HOST" \
  --port "$PORT" \
  --chunk-size "$CHUNK_SIZE"
