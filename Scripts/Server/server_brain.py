from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import zmq
from lerobot.policies.factory import make_pre_post_processors

from inference_protocol import build_error_response, build_success_response


POLICY_TYPES = ("auto", "act", "smolvla", "pi0")


@dataclass(frozen=True)
class PolicyAdapter:
    policy_type: str
    load_policy: Callable[[str, bool], Any]
    obs_to_torch: Callable[[dict[str, Any], torch.device], dict[str, Any]]
    promote_model_inputs: bool = False


def build_args(default_policy_type: str = "auto"):
    parser = argparse.ArgumentParser(description="Unified LeRobot ZMQ inference server")
    parser.add_argument("--policy-path", type=str, required=True, help="Path or HF repo id of trained policy")
    parser.add_argument(
        "--policy-type",
        type=str,
        default=default_policy_type,
        choices=POLICY_TYPES,
        help="Policy family. Use auto to infer from --policy-path.",
    )
    parser.add_argument("--bind", type=str, default="tcp://127.0.0.1:5555", help="ZMQ bind endpoint")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")
    parser.add_argument("--strict", action="store_true", help="Use strict=True when loading checkpoint")
    parser.add_argument(
        "--inspect-first-n",
        type=int,
        default=0,
        help="Print incoming observation dtype/range for the first N requests.",
    )
    return parser.parse_args()


def _infer_policy_type(policy_path: str, requested_type: str, auto_fallback: str = "act") -> str:
    if requested_type != "auto":
        return requested_type

    path_lower = policy_path.lower()
    if "smolvla" in path_lower:
        return "smolvla"
    if "pi0.5" in path_lower or "pi05" in path_lower or "pi0" in path_lower:
        return "pi0"
    return auto_fallback


def _load_act_policy(policy_path: str, strict: bool):
    from lerobot.policies.act.modeling_act import ACTPolicy

    return ACTPolicy.from_pretrained(policy_path, strict=strict)


def _load_smolvla_policy(policy_path: str, strict: bool):
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except Exception as exc:
        raise ImportError(
            "SmolVLAPolicy is unavailable in current lerobot installation. "
            "Please upgrade lerobot to a version that includes smolvla."
        ) from exc

    return SmolVLAPolicy.from_pretrained(policy_path, strict=strict)


def _load_pi0_policy(policy_path: str, strict: bool):
    try:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    except Exception as exc:
        raise ImportError(
            "PI0Policy is unavailable in current lerobot installation. "
            "Please upgrade lerobot to a version that includes pi0."
        ) from exc

    return PI0Policy.from_pretrained(policy_path, strict=strict)


def _ensure_image_uint8_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3 (CHW), got shape={image.shape}")

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        max_val = float(np.max(image)) if image.size > 0 else 1.0
        if max_val <= 1.5:
            image = np.clip(np.round(image * 255.0), 0.0, 255.0)
        else:
            image = np.clip(np.round(image), 0.0, 255.0)
        return image.astype(np.uint8, copy=False)

    image = np.clip(image.astype(np.float32), 0.0, 255.0)
    return image.astype(np.uint8, copy=False)


def _to_torch_obs_basic(obs_dict: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obs_dict.items():
        if isinstance(value, list):
            out[key] = value
        elif isinstance(value, np.ndarray):
            out[key] = torch.from_numpy(value).to(device=device)
        else:
            out[key] = value
    return out


def _to_torch_obs_vla(obs_dict: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obs_dict.items():
        if isinstance(value, list):
            out[key] = value
            continue

        if isinstance(value, np.ndarray):
            if key.startswith("observation.images."):
                value = _ensure_image_uint8_chw(value)
                out[key] = torch.from_numpy(value).to(device=device)
            elif key == "observation.state":
                out[key] = torch.from_numpy(value.astype(np.float32, copy=False)).to(device=device)
            else:
                out[key] = torch.from_numpy(value).to(device=device)
            continue

        out[key] = value

    return out


def _promote_image_like_uint8_to_float(obj: Any) -> Any:
    if torch.is_tensor(obj):
        if obj.dtype == torch.uint8 and obj.ndim >= 3:
            return obj.to(dtype=torch.float32) / 255.0
        return obj

    if isinstance(obj, dict):
        return {k: _promote_image_like_uint8_to_float(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_promote_image_like_uint8_to_float(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_promote_image_like_uint8_to_float(v) for v in obj)

    if hasattr(obj, "__dict__"):
        for attr_name, attr_value in vars(obj).items():
            try:
                setattr(obj, attr_name, _promote_image_like_uint8_to_float(attr_value))
            except Exception:
                pass
        return obj

    return obj


def _extract_action_tensor(action_obj: Any) -> torch.Tensor:
    if isinstance(action_obj, dict):
        action_tensor = action_obj.get("action")
        if action_tensor is None:
            raise ValueError("Action dict does not include 'action'.")
        return action_tensor
    if not torch.is_tensor(action_obj):
        raise ValueError(f"Unsupported action type: {type(action_obj)!r}")
    return action_obj


def _to_action_numpy(action_obj: Any) -> np.ndarray:
    action_tensor = _extract_action_tensor(action_obj)

    if action_tensor.ndim == 0:
        raise ValueError("Policy returned scalar action. Expected [A], [T, A], or [B, T, A].")

    if action_tensor.ndim == 3:
        if action_tensor.shape[0] != 1:
            raise ValueError(
                f"Unexpected batched action shape {tuple(action_tensor.shape)}. "
                "Expected batch size 1 during online inference."
            )
        action_tensor = action_tensor[0]
    elif action_tensor.ndim > 3:
        raise ValueError(
            f"Unsupported action tensor rank {action_tensor.ndim} with shape {tuple(action_tensor.shape)}"
        )

    return action_tensor.detach().to("cpu").float().numpy()


def _describe_obs(obs_torch: dict[str, Any]) -> str:
    chunks: list[str] = []
    for key in sorted(obs_torch.keys()):
        value = obs_torch[key]
        if torch.is_tensor(value):
            if value.numel() > 0 and torch.is_floating_point(value):
                v_min = float(value.min().item())
                v_max = float(value.max().item())
                chunks.append(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}, range=[{v_min:.4f}, {v_max:.4f}]")
            else:
                chunks.append(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            chunks.append(f"{key}: type={type(value).__name__}")
    return " | ".join(chunks)


def _build_adapter(policy_type: str) -> PolicyAdapter:
    if policy_type == "act":
        return PolicyAdapter(
            policy_type="act",
            load_policy=_load_act_policy,
            obs_to_torch=_to_torch_obs_basic,
            promote_model_inputs=False,
        )

    if policy_type == "smolvla":
        return PolicyAdapter(
            policy_type="smolvla",
            load_policy=_load_smolvla_policy,
            obs_to_torch=_to_torch_obs_vla,
            promote_model_inputs=True,
        )

    if policy_type == "pi0":
        return PolicyAdapter(
            policy_type="pi0",
            load_policy=_load_pi0_policy,
            obs_to_torch=_to_torch_obs_vla,
            promote_model_inputs=True,
        )

    raise ValueError(f"Unsupported policy_type: {policy_type}")


def _run_inference(policy, preprocess, postprocess, adapter: PolicyAdapter, obs_torch: dict[str, Any]) -> np.ndarray:
    model_inputs = preprocess(obs_torch)
    if adapter.promote_model_inputs:
        model_inputs = _promote_image_like_uint8_to_float(model_inputs)

    raw_action = policy.select_action(model_inputs)
    post_action = postprocess(raw_action)
    return _to_action_numpy(post_action)


def main(default_policy_type: str = "auto", auto_fallback: str = "act") -> int:
    args = build_args(default_policy_type=default_policy_type)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    policy_type = _infer_policy_type(args.policy_path, args.policy_type, auto_fallback=auto_fallback)
    adapter = _build_adapter(policy_type)
    policy = adapter.load_policy(args.policy_path, args.strict)
    policy.to(device)
    policy.eval()
    policy.reset()

    preprocess, postprocess = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"[INFO] Unified brain server listening on {args.bind}")
    print(f"[INFO] Policy type: {policy_type}")
    print(f"[INFO] Policy loaded from: {args.policy_path}")
    print(f"[INFO] Inference device: {device}")

    request_count = 0
    try:
        while True:
            request = socket.recv_pyobj()
            request_count += 1

            try:
                if not isinstance(request, dict):
                    raise TypeError(f"Expected observation dict, got {type(request).__name__}.")
                obs_dict = dict(request)

                reset_policy = bool(obs_dict.pop("reset_policy", False))
                if reset_policy:
                    policy.reset()
                    print("[INFO] Policy state reset by client request.")

                obs_torch = adapter.obs_to_torch(obs_dict, device=device)

                if request_count <= max(0, int(args.inspect_first_n)):
                    print(f"[DEBUG] request#{request_count} {_describe_obs(obs_torch)}")

                with torch.inference_mode():
                    action_np = _run_inference(policy, preprocess, postprocess, adapter, obs_torch)
                response = build_success_response(action_np.tolist())
            except Exception as exc:
                print(f"[ERROR] request#{request_count} {adapter.policy_type} inference failed: {exc}")
                try:
                    policy.reset()
                except Exception as reset_exc:
                    print(f"[WARN] Failed to reset policy after inference error: {reset_exc}")
                response = build_error_response(exc)

            socket.send_pyobj(response)
    except KeyboardInterrupt:
        print("[INFO] Brain server interrupted by user.")
    finally:
        socket.close(0)
        context.term()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
