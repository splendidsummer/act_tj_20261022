#!/usr/bin/env python3
"""Gento client for the act_tj LeRobot policy server.

This file lives in act_tj/client. openpi_client is imported from the
openpi_tj_data_collector checkout, not from this directory.

Changes from vla_client/openpi_client_gento.py on branch openpi_tj_trian_New:

- Talks to act_tj ``python -m serving.serve_policy``, not Pi05
  ``scripts/serve_policy_gento_ws.py``.
- Default policy host is 6.6.7.134:8000.
- Startup pose is off. The Pi05 preset puts waist yaw/pitch at 0, which is
  outside the ACT and Diffusion demonstrations.
- ``--replan-steps`` defaults to the server ``chunk_size``: 30 for ACT, 8 for
  Diffusion. Passing a larger value still fails the chunk check.
- Control interpolation stays at 500 Hz. The old Python default was 200 Hz;
  ``openpi_gento_client.sh`` already overrode it to 500.
- ``--image-size`` is still sent and is still ignored. The LeRobot server
  splits the undistorted quad into 960x744 tiles, then blacks out the bottom
  48 rows of the two wrist images. It does not resize to 224.
- Image topic stays ``/quad_tile/compressed_undistorted``.
- Prompt is forwarded but these checkpoints have no language input. The server
  does not condition on it.
- Refuses a server whose metadata ``policy`` is neither ``act`` nor
  ``diffusion``, and refuses a server that does not return absolute actions.
- Drops the unused ``pdb`` breakpoint.

State order, waist swap, ``/tj`` command topics, and absolute-action
interpolation are unchanged.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import pathlib
import signal
import statistics
import sys
import time

import numpy as np
import rclpy
from marvin_msgs.msg import JointcmdArm
from marvin_msgs.msg import JointcmdBody
from marvin_msgs.msg import JointcmdHead
from marvin_msgs.srv import Int
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray

_openpi_client_candidates = (
    pathlib.Path("/home/holdon/Projects/openpi_tj_data_collector/packages/openpi-client/src"),
    pathlib.Path("/home/nvidia/Projects/openpi_tj_data_collector/packages/openpi-client/src"),
    pathlib.Path(__file__).resolve().parents[1] / "packages" / "openpi-client" / "src",
)
openpi_client_src = next((path for path in _openpi_client_candidates if path.is_dir()), None)
if openpi_client_src is not None and str(openpi_client_src) not in sys.path:
    sys.path.insert(0, str(openpi_client_src))

DEFAULT_POLICY_HOST = "6.6.7.134"
DEFAULT_POLICY_PORT = 8000
DEFAULT_TASK_PROMPT = "remove the object"
DEFAULT_IMAGE_TOPIC = "/quad_tile/compressed_undistorted"
DEFAULT_QUAD_WIDTH = 1920
DEFAULT_QUAD_HEIGHT = 1488
DEFAULT_STARTUP_STATE = (
    -1.16,
    -0.57,
    1.13,
    -1.24,
    -1.61,
    0.52,
    -0.57,
    0,
    1.16,
    -0.56,
    -1.14,
    -1.25,
    1.61,
    0.54,
    0.56,
    0,
    0.54,
    0.0,
    0.0,
    0.0,
    0.0,
)
GENTO_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
GENTO_STATE_DIM = 21
GENTO_ACTION_DIM = 21
GENTO_ACTION_NAMES = (
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
JOINT_COUNT = 19
LEFT_ARM = slice(0, 7)
RIGHT_ARM = slice(7, 14)
LIFT_INDEX = 14
BODY = slice(15, 17)
HEAD = slice(17, 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 Gento client for the act_tj ACT or Diffusion policy server.")
    parser.add_argument("--policy-host", default=DEFAULT_POLICY_HOST, help="Policy server IP. GPU machine is 6.6.7.134.")
    parser.add_argument("--policy-port", type=int, default=DEFAULT_POLICY_PORT, help="Policy server WebSocket port.")
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--action-rate", type=float, default=30.0, help="Policy action keyframe rate in Hz.")
    parser.add_argument(
        "--control-rate", type=float, default=500.0, help="Interpolated ROS command publish rate in Hz."
    )
    parser.add_argument(
        "--replan-steps",
        type=int,
        default=None,
        help="Actions to execute per chunk. Default is the server chunk_size (ACT 30, Diffusion 8).",
    )
    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC, help="Compressed quad image topic.")
    parser.add_argument("--quad-width", type=int, default=DEFAULT_QUAD_WIDTH, help="Decoded quad image width.")
    parser.add_argument("--quad-height", type=int, default=DEFAULT_QUAD_HEIGHT, help="Decoded quad image height.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Sent in the observation and ignored. The LeRobot server uses 960x744 tiles.",
    )
    parser.add_argument("--disable-action-post", action="store_true", help="Inference only; do not control robot.")
    parser.add_argument("--set-custom-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hold-seconds", type=float, default=1.0, help="Publish hold commands before Custom input.")
    parser.add_argument(
        "--startup-pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move to the Pi05 preset pose first. Leave this off for ACT and Diffusion.",
    )
    parser.add_argument(
        "--startup-pose-seconds", type=float, default=3.0, help="Seconds to publish the preset startup pose."
    )
    parser.add_argument("--state-timeout", type=float, default=20.0, help="Seconds to wait for first ROS observation.")
    parser.add_argument("--max-angular-step", type=float, default=0.12, help="Per-step angular command clamp in rad.")
    parser.add_argument("--max-lift-step", type=float, default=0.01, help="Per-step lift command clamp in meters.")
    parser.add_argument(
        "--max-gripper-step", type=float, default=0.2, help="Per-step gripper command clamp in 0..1 units."
    )
    parser.add_argument("--print-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timing-log-seconds", type=float, default=2.0, help="Timing summary interval in seconds.")
    parser.add_argument("--cv-show-inputs", action="store_true")
    parser.add_argument("--cv-show-wait-ms", type=int, default=1)
    return parser.parse_args()


def clipped_gripper(feedback: Float32MultiArray | None) -> float:
    if feedback is None or len(feedback.data) == 0:
        return 0.0
    return float(np.clip(float(feedback.data[0]), 0.0, 1.0))


def h264_contains_idr(payload: bytes) -> bool:
    index = 0
    while index + 4 < len(payload):
        if payload[index : index + 4] == b"\x00\x00\x00\x01":
            if payload[index + 4] & 0x1F == 5:
                return True
            index += 4
        elif payload[index : index + 3] == b"\x00\x00\x01":
            if payload[index + 3] & 0x1F == 5:
                return True
            index += 3
        else:
            index += 1
    return False


def clamp_toward(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    delta = np.clip(delta, -max_step, max_step)
    return np.asarray(current, dtype=np.float64) + delta


def interpolate_actions(
    start: np.ndarray, actions: np.ndarray, action_rate: float, control_rate: float
) -> list[np.ndarray]:
    if action_rate <= 0.0 or control_rate <= 0.0:
        raise ValueError("action and control rates must be positive")
    if control_rate < action_rate:
        raise ValueError("control rate must be at least the policy action rate")

    start = np.asarray(start, dtype=np.float64).reshape(-1)
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != start.shape[0]:
        raise ValueError(f"cannot interpolate start shape {start.shape} with actions shape {actions.shape}")

    ratio = control_rate / action_rate
    interpolated: list[np.ndarray] = []
    previous = start
    for index, target in enumerate(actions):
        segment_start = round(index * ratio)
        segment_end = round((index + 1) * ratio)
        sample_count = segment_end - segment_start
        for sample in range(1, sample_count + 1):
            alpha = sample / sample_count
            interpolated.append(previous + alpha * (target - previous))
        previous = target
    return interpolated


def validate_server_metadata(metadata: dict, action_rate: float, replan_steps: int) -> None:
    if "action_dim" in metadata and int(metadata["action_dim"]) != GENTO_ACTION_DIM:
        raise RuntimeError(
            f"server action_dim={metadata['action_dim']} does not match client action_dim={GENTO_ACTION_DIM}"
        )
    if "output_action_schema" in metadata:
        server_schema = tuple(metadata["output_action_schema"])
        if server_schema != GENTO_ACTION_NAMES:
            raise RuntimeError(f"server output_action_schema does not match client schema: {server_schema}")
    if "action_rate" in metadata and not np.isclose(float(metadata["action_rate"]), action_rate):
        raise RuntimeError(
            f"server action_rate={metadata['action_rate']} does not match client action_rate={action_rate}"
        )
    if "chunk_size" in metadata and int(metadata["chunk_size"]) < max(1, replan_steps):
        raise RuntimeError(
            f"server chunk_size={metadata['chunk_size']} is smaller than client replan_steps={replan_steps}"
        )
    policy_name = metadata.get("policy")
    if policy_name is not None and policy_name not in {"act", "diffusion"}:
        raise RuntimeError(
            f"expected the act_tj LeRobot server, got policy={policy_name!r}. "
            "Do not connect this client to scripts/serve_policy_gento_ws.py."
        )
    action_mode = metadata.get("action_mode")
    if action_mode is not None and action_mode != "absolute":
        raise RuntimeError(f"server action_mode={action_mode!r}; expected absolute joint targets")


class TimingMonitor:
    def __init__(self, log_interval: float) -> None:
        self.log_interval = log_interval
        self.samples: dict[str, collections.deque[float]] = collections.defaultdict(
            lambda: collections.deque(maxlen=2000)
        )
        self.gauges: dict[str, float] = {}
        self.last_log = time.monotonic()
        self.publish_count = 0
        self.publish_window_start = self.last_log

    def add(self, name: str, value_ms: float) -> None:
        self.samples[name].append(float(value_ms))

    def set_gauge(self, name: str, value: float) -> None:
        self.gauges[name] = float(value)

    def published(self) -> None:
        self.publish_count += 1

    def summary(self) -> str | None:
        now = time.monotonic()
        if now - self.last_log < self.log_interval:
            return None
        elapsed = now - self.publish_window_start
        fields = [f"publish_hz={self.publish_count / elapsed:.1f}"]
        fields.extend(f"{name}={value:.1f}" for name, value in sorted(self.gauges.items()))
        for name in sorted(self.samples):
            values = list(self.samples[name])
            if not values:
                continue
            percentile_index = min(len(values) - 1, int(0.95 * len(values)))
            p95 = sorted(values)[percentile_index]
            fields.append(f"{name}_ms={statistics.fmean(values):.2f}/{p95:.2f}/{max(values):.2f}")
        self.samples.clear()
        self.publish_count = 0
        self.publish_window_start = now
        self.last_log = now
        return "[timing] avg/p95/max " + " ".join(fields)


class GentoRosClient(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("openpi_client_gento")
        self.args = args
        self.joint_state: JointState | None = None
        self.left_gripper_feedback: Float32MultiArray | None = None
        self.right_gripper_feedback: Float32MultiArray | None = None
        self.compressed_image: bytes | None = None
        self.compressed_image_packets: collections.deque[bytes] = collections.deque()
        self.h264_stream_ready = False
        self.last_image_format: str | None = None
        self.last_image_error: str | None = None
        self.last_body_command: np.ndarray | None = None
        self.last_head_command: np.ndarray | None = None
        self.last_lift_command: float | None = None
        self.last_arm_a_command: np.ndarray | None = None
        self.last_arm_b_command: np.ndarray | None = None
        self.last_left_gripper_command = 0.0
        self.last_right_gripper_command = 0.0
        self.last_joint_time: float | None = None
        self.last_left_gripper_time: float | None = None
        self.last_right_gripper_time: float | None = None
        self.last_image_time: float | None = None
        self.last_image_process_ms = 0.0

        self.create_subscription(JointState, "/tj/joint_states", self._joint_state_callback, qos_profile_sensor_data)
        self.create_subscription(
            Float32MultiArray, "/tj/info/gripper_feedback_L", self._left_gripper_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            Float32MultiArray, "/tj/info/gripper_feedback_R", self._right_gripper_callback, qos_profile_sensor_data
        )
        self.create_subscription(CompressedImage, args.image_topic, self._image_callback, qos_profile_sensor_data)

        self.arm_a_publisher = self.create_publisher(JointcmdArm, "/tj/control/user/joint_cmd_A", 10)
        self.arm_b_publisher = self.create_publisher(JointcmdArm, "/tj/control/user/joint_cmd_B", 10)
        self.body_publisher = self.create_publisher(JointcmdBody, "/tj/control/user/joint_cmd_body", 10)
        self.head_publisher = self.create_publisher(JointcmdHead, "/tj/control/user/joint_cmd_head", 10)
        self.left_gripper_publisher = self.create_publisher(Float32, "/tj/control/gripperValueL", 10)
        self.right_gripper_publisher = self.create_publisher(Float32, "/tj/control/gripperValueR", 10)
        self.set_input_client = self.create_client(Int, "/tj/control/set_input")

    def _joint_state_callback(self, message: JointState) -> None:
        self.joint_state = message
        self.last_joint_time = time.monotonic()

    def _left_gripper_callback(self, message: Float32MultiArray) -> None:
        self.left_gripper_feedback = message
        self.last_left_gripper_time = time.monotonic()

    def _right_gripper_callback(self, message: Float32MultiArray) -> None:
        self.right_gripper_feedback = message
        self.last_right_gripper_time = time.monotonic()

    def _image_callback(self, message: CompressedImage) -> None:
        process_start = time.perf_counter()
        self.last_image_format = message.format
        payload = bytes(message.data)
        self.compressed_image = payload
        image_format = message.format.lower()
        if any(codec in image_format for codec in ("h264", "avc", "x264")):
            if h264_contains_idr(payload):
                self.compressed_image_packets.clear()
                self.h264_stream_ready = True
            if self.h264_stream_ready:
                self.compressed_image_packets.append(payload)
        else:
            self.compressed_image_packets.clear()
            self.compressed_image_packets.append(payload)
            self.h264_stream_ready = True
        self.last_image_error = None
        self.last_image_time = time.monotonic()
        self.last_image_process_ms = (time.perf_counter() - process_start) * 1000.0

    def observation_ages_ms(self) -> dict[str, float]:
        now = time.monotonic()
        timestamps = {
            "joint_age": self.last_joint_time,
            "left_gripper_age": self.last_left_gripper_time,
            "right_gripper_age": self.last_right_gripper_time,
            "image_age": self.last_image_time,
        }
        return {name: (now - stamp) * 1000.0 for name, stamp in timestamps.items() if stamp is not None}

    def ready(self) -> bool:
        return self.joint_state is not None and self.h264_stream_ready and bool(self.compressed_image_packets)

    def missing_observations(self) -> list[str]:
        missing = []
        if self.joint_state is None:
            missing.append("/tj/joint_states")
        if self.compressed_image is None:
            detail = self.args.image_topic
            if self.last_image_format is not None:
                detail += f" format={self.last_image_format!r}"
            if self.last_image_error is not None:
                detail += f" ({self.last_image_error})"
            missing.append(detail)
        return missing

    def wait_for_observation(self) -> bool:
        start = time.monotonic()
        while rclpy.ok() and time.monotonic() - start < self.args.state_timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.ready():
                return True
        return False

    def current_state21(self) -> np.ndarray:
        if self.joint_state is None or len(self.joint_state.position) < JOINT_COUNT:
            raise RuntimeError("/tj/joint_states is missing or shorter than 19 positions")
        positions = np.asarray(self.joint_state.position[:JOINT_COUNT], dtype=np.float32)
        state = np.zeros(GENTO_STATE_DIM, dtype=np.float32)
        state[0:7] = positions[LEFT_ARM]
        state[7] = clipped_gripper(self.left_gripper_feedback)
        state[8:15] = positions[RIGHT_ARM]
        state[15] = clipped_gripper(self.right_gripper_feedback)
        state[16] = positions[LIFT_INDEX]
        state[17] = positions[BODY][1]  # waist_yaw
        state[18] = positions[BODY][0]  # waist_pitch
        state[19] = positions[HEAD][0]  # head_yaw
        state[20] = positions[HEAD][1]  # head_pitch
        return state

    def observation(self) -> dict:
        if not self.compressed_image_packets:
            raise RuntimeError(f"{self.args.image_topic} has not produced a compressed image yet")
        packets = list(self.compressed_image_packets)
        self.compressed_image_packets.clear()
        return {
            "compressed_image": {
                "packets": packets,
                "format": self.last_image_format or "",
                "quad_width": self.args.quad_width,
                "quad_height": self.args.quad_height,
                "image_size": self.args.image_size,
            },
            "state": self.current_state21(),
            "prompt": self.args.task_prompt,
        }

    def set_input(self, mode: int, timeout: float = 5.0) -> bool:
        if not self.set_input_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("/tj/control/set_input is unavailable")
            return False
        request = Int.Request()
        request.data = mode
        future = self.set_input_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if not future.done() or future.exception() is not None:
            self.get_logger().error(f"set_input({mode}) failed or timed out")
            return False
        response = future.result()
        self.get_logger().info(
            f"set_input({mode}): success={response.success}, error_code={response.error_code}, message={response.message}"
        )
        return bool(response.success)

    def publish_hold(self) -> None:
        state = self.current_state21()
        self.publish_action(state)

    def current_command21(self) -> np.ndarray:
        if self.last_arm_a_command is None:
            return self.current_state21().astype(np.float64)
        return np.concatenate(
            [
                self.last_arm_a_command,
                [self.last_left_gripper_command],
                self.last_arm_b_command,
                [self.last_right_gripper_command, self.last_lift_command],
                self.last_body_command,
                self.last_head_command,
            ]
        )

    def publish_action(self, action21: np.ndarray) -> None:
        if self.joint_state is None or len(self.joint_state.position) < JOINT_COUNT:
            raise RuntimeError("cannot publish action without /tj/joint_states")
        action = np.asarray(action21, dtype=np.float64).reshape(-1)
        if action.shape[0] < GENTO_ACTION_DIM:
            raise ValueError(f"expected at least {GENTO_ACTION_DIM} action values, got {action.shape}")

        current = np.asarray(self.joint_state.position[:JOINT_COUNT], dtype=np.float64)
        arm_a_target = action[0:7]
        arm_b_target = action[8:15]
        lift_target = float(action[16])
        body_target = action[17:19]
        head_target = action[19:21]
        left_gripper_target = float(np.clip(action[7], 0.0, 1.0))
        right_gripper_target = float(np.clip(action[15], 0.0, 1.0))

        if self.last_arm_a_command is None:
            self.last_arm_a_command = current[LEFT_ARM].copy()
            self.last_arm_b_command = current[RIGHT_ARM].copy()
            self.last_body_command = np.asarray([current[BODY][1], current[BODY][0]], dtype=np.float64)
            self.last_head_command = current[HEAD].copy()
            self.last_lift_command = float(current[LIFT_INDEX])
            self.last_left_gripper_command = clipped_gripper(self.left_gripper_feedback)
            self.last_right_gripper_command = clipped_gripper(self.right_gripper_feedback)

        self.last_arm_a_command = clamp_toward(self.last_arm_a_command, arm_a_target, self.args.max_angular_step)
        self.last_arm_b_command = clamp_toward(self.last_arm_b_command, arm_b_target, self.args.max_angular_step)
        self.last_lift_command = float(
            clamp_toward(np.array([self.last_lift_command]), np.array([lift_target]), self.args.max_lift_step)[0]
        )
        self.last_head_command = clamp_toward(self.last_head_command, head_target, self.args.max_angular_step)
        self.last_body_command = clamp_toward(self.last_body_command, body_target, self.args.max_angular_step)
        self.last_left_gripper_command = float(
            clamp_toward(
                np.array([self.last_left_gripper_command]), np.array([left_gripper_target]), self.args.max_gripper_step
            )[0]
        )
        self.last_right_gripper_command = float(
            clamp_toward(
                np.array([self.last_right_gripper_command]),
                np.array([right_gripper_target]),
                self.args.max_gripper_step,
            )[0]
        )

        stamp = self.get_clock().now().to_msg()
        arm_a = JointcmdArm()
        arm_a.header.stamp = stamp
        arm_a.positions = [float(x) for x in self.last_arm_a_command]
        self.arm_a_publisher.publish(arm_a)

        arm_b = JointcmdArm()
        arm_b.header.stamp = stamp
        arm_b.positions = [float(x) for x in self.last_arm_b_command]
        self.arm_b_publisher.publish(arm_b)

        body = JointcmdBody()
        body.header.stamp = stamp
        # Policy order is [waist_yaw, waist_pitch], while JointcmdBody expects
        # [lift, waist_pitch, waist_yaw, ...].
        body.positions = [
            float(self.last_lift_command),
            float(self.last_body_command[1]),
            float(self.last_body_command[0]),
            0.0,
            0.0,
            0.0,
        ]
        self.body_publisher.publish(body)

        head = JointcmdHead()
        head.header.stamp = stamp
        head.positions = [float(self.last_head_command[0]), float(self.last_head_command[1]), 0.0]
        self.head_publisher.publish(head)

        left_gripper = Float32()
        left_gripper.data = self.last_left_gripper_command
        self.left_gripper_publisher.publish(left_gripper)

        right_gripper = Float32()
        right_gripper.data = self.last_right_gripper_command
        self.right_gripper_publisher.publish(right_gripper)


def main() -> None:
    args = parse_args()
    if args.action_rate <= 0.0:
        raise ValueError("--action-rate must be positive")
    if args.control_rate < args.action_rate:
        raise ValueError("--control-rate must be at least --action-rate")
    if args.timing_log_seconds <= 0.0:
        raise ValueError("--timing-log-seconds must be positive")
    if args.startup_pose_seconds < 0.0:
        raise ValueError("--startup-pose-seconds must be non-negative")
    if args.cv_show_inputs:
        raise ValueError("--cv-show-inputs is unavailable when image decoding runs on the policy server")
    rclpy.init()
    node = GentoRosClient(args)
    timing = TimingMonitor(args.timing_log_seconds)
    action_plan: collections.deque[np.ndarray] = collections.deque()
    policy_executor: concurrent.futures.ThreadPoolExecutor | None = None
    interrupted = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        node.get_logger().info(f"waiting for /tj/joint_states and {args.image_topic}")
        if not node.wait_for_observation():
            missing = ", ".join(node.missing_observations())
            raise RuntimeError(f"timed out waiting for ROS observation; missing: {missing}")

        if not args.disable_action_post:
            hold_end = time.monotonic() + args.hold_seconds
            while time.monotonic() < hold_end and not interrupted:
                rclpy.spin_once(node, timeout_sec=0.0)
                node.publish_hold()
                time.sleep(1.0 / args.control_rate)
            if args.set_custom_input and not node.set_input(3):
                raise RuntimeError("failed to switch /tj/control/set_input to Custom")
            if args.startup_pose and args.startup_pose_seconds > 0.0:
                startup_action = np.asarray(DEFAULT_STARTUP_STATE, dtype=np.float64)
                if startup_action.shape != (GENTO_ACTION_DIM,):
                    raise RuntimeError(f"invalid startup pose shape: {startup_action.shape}")
                startup_end = time.monotonic() + args.startup_pose_seconds
                node.get_logger().info(
                    f"moving to startup pose for {args.startup_pose_seconds:.1f}s before policy inference"
                )
                while time.monotonic() < startup_end and not interrupted:
                    rclpy.spin_once(node, timeout_sec=0.0)
                    node.publish_action(startup_action)
                    time.sleep(1.0 / args.control_rate)

        if interrupted or not rclpy.ok():
            return

        from openpi_client import websocket_client_policy

        policy = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
        server_metadata = policy.get_server_metadata()
        if args.replan_steps is None:
            if "chunk_size" not in server_metadata:
                raise RuntimeError("pass --replan-steps; server metadata has no chunk_size")
            args.replan_steps = int(server_metadata["chunk_size"])
        validate_server_metadata(server_metadata, args.action_rate, args.replan_steps)
        policy_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-inference")
        node.get_logger().info(
            f"connected to policy server {args.policy_host}:{args.policy_port}; "
            f"policy={server_metadata.get('policy', 'unknown')} "
            f"server={server_metadata.get('server', 'unknown')} "
            f"chunk_size={server_metadata.get('chunk_size', 'unknown')} "
            f"replan_steps={args.replan_steps} "
            f"ckpt={server_metadata.get('ckpt', 'unknown')}"
        )

        next_tick = time.monotonic()
        infer_count = 0
        while rclpy.ok() and not interrupted:
            spin_start = time.perf_counter()
            rclpy.spin_once(node, timeout_sec=0.0)
            timing.add("ros_spin", (time.perf_counter() - spin_start) * 1000.0)
            if not action_plan:
                observation_start = time.perf_counter()
                observation = node.observation()
                timing.add("observation_build", (time.perf_counter() - observation_start) * 1000.0)
                timing.add("image_process", node.last_image_process_ms)
                for name, age_ms in node.observation_ages_ms().items():
                    timing.add(name, age_ms)
                state = observation["state"]
                if args.print_input:
                    compressed = observation["compressed_image"]
                    payload_size = sum(len(packet) for packet in compressed["packets"])
                    node.get_logger().info(
                        f"obs compressed={payload_size / 1024.0:.1f}KiB packets={len(compressed['packets'])} "
                        f"format={compressed['format']!r} state[{len(state)}]={np.array2string(state, precision=3)}"
                    )
                inference = policy_executor.submit(policy.infer, observation)
                inference_tick = time.monotonic()
                while not inference.done() and rclpy.ok() and not interrupted:
                    spin_start = time.perf_counter()
                    rclpy.spin_once(node, timeout_sec=0.0)
                    timing.add("ros_spin", (time.perf_counter() - spin_start) * 1000.0)
                    if not args.disable_action_post:
                        publish_start = time.perf_counter()
                        node.publish_action(node.current_command21())
                        timing.add("publish", (time.perf_counter() - publish_start) * 1000.0)
                        timing.published()
                    inference_tick += 1.0 / args.control_rate
                    time.sleep(max(0.0, inference_tick - time.monotonic()))
                    timing.add("deadline_late", max(0.0, time.monotonic() - inference_tick) * 1000.0)
                if interrupted or not rclpy.ok():
                    break
                result = inference.result()
                for name, value in policy.get_last_timing().items():
                    if name.endswith("_kib"):
                        timing.set_gauge(f"ws_{name}", value)
                    else:
                        timing.add(f"ws_{name.removesuffix('_ms')}", value)
                server_timing = result.get("server_timing", {})
                if "image_decode_ms" in result:
                    timing.add("server_image_decode", float(result["image_decode_ms"]))
                if "infer_ms" in server_timing:
                    timing.add("server_infer", float(server_timing["infer_ms"]))
                    websocket_timing = policy.get_last_timing()
                    communication_ms = (
                        websocket_timing.get("roundtrip_ms", 0.0)
                        - websocket_timing.get("pack_ms", 0.0)
                        - websocket_timing.get("unpack_ms", 0.0)
                        - float(server_timing["infer_ms"])
                    )
                    timing.add("communication_residual", max(0.0, communication_ms))
                actions = np.asarray(result.get("actions"), dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] < GENTO_ACTION_DIM:
                    raise RuntimeError(f"invalid action chunk shape: {actions.shape}")
                source_actions = actions[: max(1, args.replan_steps), :GENTO_ACTION_DIM]
                interpolation_start = node.current_command21()
                interpolation_start_time = time.perf_counter()
                action_plan.extend(
                    interpolate_actions(interpolation_start, source_actions, args.action_rate, args.control_rate)
                )
                timing.add("interpolation", (time.perf_counter() - interpolation_start_time) * 1000.0)
                infer_count += 1
                if args.print_output:
                    node.get_logger().info(
                        f"infer#{infer_count} actions={actions.shape} interpolated={len(action_plan)} "
                        f"first={np.array2string(actions[0, :GENTO_ACTION_DIM], precision=3)}"
                    )
                next_tick = time.monotonic()
            action = action_plan.popleft()
            if not args.disable_action_post:
                publish_start = time.perf_counter()
                node.publish_action(action)
                timing.add("publish", (time.perf_counter() - publish_start) * 1000.0)
                timing.published()

            next_tick += 1.0 / args.control_rate
            time.sleep(max(0.0, next_tick - time.monotonic()))
            timing.add("deadline_late", max(0.0, time.monotonic() - next_tick) * 1000.0)
            if args.timing:
                timing_summary = timing.summary()
                if timing_summary is not None:
                    node.get_logger().info(timing_summary)
    except KeyboardInterrupt:
        node.get_logger().info("shutdown requested")
    finally:
        if policy_executor is not None:
            policy_executor.shutdown(wait=False, cancel_futures=True)
        if not args.disable_action_post:
            with contextlib.suppress(Exception):
                node.set_input(0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
