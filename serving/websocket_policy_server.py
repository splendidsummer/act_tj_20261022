"""Websocket policy server.

Protocol matches OpenPI's WebsocketPolicyServer:
on connect the server sends msgpack metadata, then each binary observation
is answered with a binary action dict. HTTP GET /healthz returns OK.
"""

import asyncio
import http
import logging
import time
import traceback

import websockets
import websockets.asyncio.server as _server
import websockets.frames

from serving.msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000, metadata: dict | None = None) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._infer_lock = asyncio.Lock()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info(
                "%s websocket server on ws://%s:%s",
                self._metadata.get("policy", "policy"),
                self._host,
                self._port,
            )
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        self._policy.reset()
        packer = Packer()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = unpackb(await websocket.recv())
                infer_time = time.monotonic()
                async with self._infer_lock:
                    action = await asyncio.to_thread(self._policy.infer, obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
