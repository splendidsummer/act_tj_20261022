"""Decode the undistorted quad image sent by the Gento client.

Layout matches the recorder: top-left cam_high, bottom-left cam_left_wrist,
bottom-right cam_right_wrist. Each tile is resized to the training size,
960x744. The client's image_size field is ignored, so the tiles are not
resized to 224. After that resize, the bottom 48 rows of the two wrist
tiles are set to black. Training frames of FlipAndRemove and MoveParcel
have that same 48-row bar. cam_high is left unchanged.
"""

import time

import av
import cv2
import numpy as np

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
WRIST_CAMERAS = ("cam_left_wrist", "cam_right_wrist")
TILE_WIDTH = 960
TILE_HEIGHT = 744
WRIST_BOTTOM_PX = 48


class QuadImageDecoder:
    def __init__(self) -> None:
        av.logging.set_level(av.logging.PANIC)
        self._h264_decoder = av.CodecContext.create("h264", "r")
        self._last_frame_rgb: np.ndarray | None = None

    def decode(self, compressed: dict) -> tuple[dict[str, np.ndarray], float]:
        start = time.perf_counter()
        packets = compressed.get("packets")
        if (
            not isinstance(packets, list)
            or not packets
            or not all(isinstance(packet, (bytes, bytearray)) and packet for packet in packets)
        ):
            raise ValueError("compressed_image.packets must be a non-empty list of non-empty bytes")
        image_format = str(compressed.get("format", "")).lower()
        quad_rgb = self._decode_quad_rgb([bytes(packet) for packet in packets], image_format)
        images = self._split(quad_rgb)
        return images, (time.perf_counter() - start) * 1000.0

    def _decode_quad_rgb(self, packets: list[bytes], image_format: str) -> np.ndarray:
        if any(codec in image_format for codec in ("jpeg", "jpg", "png")):
            quad_bgr = cv2.imdecode(np.frombuffer(packets[-1], dtype=np.uint8), cv2.IMREAD_COLOR)
            if quad_bgr is None:
                raise ValueError(f"OpenCV failed to decode compressed image format {image_format!r}")
            return cv2.cvtColor(quad_bgr, cv2.COLOR_BGR2RGB)

        if any(codec in image_format for codec in ("h264", "avc", "x264")):
            latest_frame = None
            try:
                for payload in packets:
                    for packet in self._h264_decoder.parse(payload):
                        frames = self._h264_decoder.decode(packet)
                        if frames:
                            latest_frame = frames[-1]
            except av.FFmpegError as error:
                raise ValueError(f"FFmpeg failed to decode H264 image: {error}") from error
            if latest_frame is not None:
                self._last_frame_rgb = latest_frame.to_ndarray(format="rgb24")
            elif self._last_frame_rgb is None:
                raise ValueError("H264 packet batch produced no frame; the stream is waiting for an IDR frame")
            return self._last_frame_rgb

        raise ValueError(f"unsupported compressed image format {image_format!r}; expected JPEG, PNG, or H264/AVC")

    @staticmethod
    def _split(quad_rgb: np.ndarray) -> dict[str, np.ndarray]:
        height, width = quad_rgb.shape[:2]
        half_height, half_width = height // 2, width // 2
        tiles = {
            "cam_high": quad_rgb[:half_height, :half_width],
            "cam_left_wrist": quad_rgb[half_height:, :half_width],
            "cam_right_wrist": quad_rgb[half_height:, half_width:],
        }
        images = {}
        for name, tile in tiles.items():
            if tile.shape[0] != TILE_HEIGHT or tile.shape[1] != TILE_WIDTH:
                tile = cv2.resize(tile, (TILE_WIDTH, TILE_HEIGHT), interpolation=cv2.INTER_AREA)
            tile = np.ascontiguousarray(tile, dtype=np.uint8)
            if name in WRIST_CAMERAS:
                if WRIST_BOTTOM_PX >= tile.shape[0]:
                    raise ValueError("wrist bottom blackout must be smaller than the wrist image height")
                tile = tile.copy()
                tile[-WRIST_BOTTOM_PX:, :] = 0
            images[name] = tile
        return images
