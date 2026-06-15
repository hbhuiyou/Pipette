from __future__ import annotations

from typing import Any


class PolicyServerError(RuntimeError):
    def __init__(self, error_type: str, message: str, retryable: bool = True):
        self.error_type = str(error_type or "InferenceError")
        self.retryable = bool(retryable)
        super().__init__(str(message or "Remote policy inference failed."))


def build_success_response(action: Any) -> dict[str, Any]:
    return {"ok": True, "action": action}


def build_error_response(exc: BaseException, *, retryable: bool = True) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": bool(retryable),
        },
    }


def extract_action_response(response: Any) -> Any:
    # Accept legacy list responses while clients and servers are upgraded together.
    if not isinstance(response, dict) or "ok" not in response:
        return response

    if bool(response.get("ok")):
        if "action" not in response:
            raise PolicyServerError("ProtocolError", "Successful response does not include an action.", False)
        return response["action"]

    error = response.get("error")
    if not isinstance(error, dict):
        raise PolicyServerError("InferenceError", "Remote policy inference failed without error details.")
    raise PolicyServerError(
        str(error.get("type") or "InferenceError"),
        str(error.get("message") or "Remote policy inference failed."),
        bool(error.get("retryable", True)),
    )
