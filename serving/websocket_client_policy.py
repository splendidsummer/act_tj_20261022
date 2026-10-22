"""Sync client for the ACT websocket server. Same message order as OpenPI."""

import logging
import time

import websockets.sync.client

from serving.msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


class WebsocketClientPolicy:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._packer = Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self):
        logger.info("Waiting for server at %s", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(1)

    def infer(self, obs: dict) -> dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpackb(response)

    def reset(self) -> dict:
        return self.infer({"reset": True})
