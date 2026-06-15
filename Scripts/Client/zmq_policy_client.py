from __future__ import annotations

from typing import Any

import zmq

from inference_protocol import extract_action_response


class RecoveringPolicyClient:
    """REQ client that replaces its socket after every transport failure."""

    def __init__(
        self,
        endpoint: str,
        timeout_ms: int,
        *,
        context: zmq.Context | None = None,
    ):
        self.endpoint = str(endpoint)
        self.timeout_ms = max(1, int(timeout_ms))
        self._owns_context = context is None
        self._context = context if context is not None else zmq.Context()
        self._socket = self._create_socket()

    def _create_socket(self):
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        return socket

    def _replace_socket(self) -> None:
        old_socket = self._socket
        self._socket = None
        if old_socket is not None:
            old_socket.close(0)
        self._socket = self._create_socket()

    def request_action(self, observation: dict[str, Any]) -> Any:
        try:
            self._socket.send_pyobj(observation)
            response = self._socket.recv_pyobj()
        except zmq.ZMQError:
            self._replace_socket()
            raise
        return extract_action_response(response)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        if self._owns_context:
            self._context.term()
