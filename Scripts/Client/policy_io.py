from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class PolicyFeatureKeys:
    top: str = "observation.images.top"
    main: str = "observation.images.main"
    wrist: str = "observation.images.wrist"
    state: str = "observation.state"


def find_gripper_probe_body_ids(robot) -> list[int]:
    body_ids: list[int] = []
    for body_name in ("panda_leftfinger", "panda_rightfinger", "panda_hand"):
        try:
            found = robot.find_bodies(body_name)[0]
        except Exception:
            found = []
        for body_id in found:
            body_id_int = int(body_id)
            if body_id_int not in body_ids:
                body_ids.append(body_id_int)
    return body_ids


def get_gripper_probe_positions(robot, body_ids: list[int]) -> torch.Tensor | None:
    if not body_ids:
        return None
    body_pos = robot.data.body_pos_w[:, body_ids]
    probe_pos = [body_pos[:, index, :] for index in range(body_pos.shape[1])]
    if body_pos.shape[1] > 1:
        probe_pos.append(body_pos.mean(dim=1))
    return torch.cat(probe_pos, dim=0)


def rgb_to_uint8_chw_numpy(frame: torch.Tensor | None) -> np.ndarray | None:
    if frame is None or frame.ndim != 3:
        return None

    rgb = frame[..., :3]
    if rgb.dtype == torch.uint8:
        rgb_u8 = rgb
    elif torch.is_floating_point(rgb):
        max_val = float(rgb.max().item()) if rgb.numel() > 0 else 1.0
        if max_val <= 1.5:
            rgb_u8 = torch.clamp(torch.round(rgb * 255.0), 0.0, 255.0).to(dtype=torch.uint8)
        else:
            rgb_u8 = torch.clamp(torch.round(rgb), 0.0, 255.0).to(dtype=torch.uint8)
    else:
        rgb_u8 = torch.clamp(rgb.to(dtype=torch.float32), 0.0, 255.0).to(dtype=torch.uint8)

    return rgb_u8.permute(2, 0, 1).contiguous().cpu().numpy()


def rgb_to_float_chw_numpy(frame: torch.Tensor | None) -> np.ndarray | None:
    if frame is None or frame.ndim != 3:
        return None

    rgb = frame[..., :3].to(dtype=torch.float32)
    rgb = torch.clamp(rgb, 0.0, 255.0) / 255.0
    return rgb.permute(2, 0, 1).contiguous().cpu().numpy().astype(np.float32, copy=False)


def build_request_observation(
    *,
    top_rgb: np.ndarray,
    main_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    robot_state: np.ndarray,
    task: str,
    reset_policy: bool,
    feature_keys: PolicyFeatureKeys,
    language_feature_mode: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {
        feature_keys.top: top_rgb,
        feature_keys.main: main_rgb,
        feature_keys.wrist: wrist_rgb,
        feature_keys.state: robot_state,
        "reset_policy": bool(reset_policy),
    }

    if language_feature_mode in ("task", "both"):
        obs["task"] = [task]
    if language_feature_mode in ("language_instruction", "both"):
        obs["language_instruction"] = [task]
    return obs


def build_safe_hold_action(robot, arm_joint_ids, device: torch.device) -> torch.Tensor:
    arm_now = robot.data.joint_pos[:, arm_joint_ids][0].to(device=device, dtype=torch.float32)
    grip_open = torch.tensor([1.0], device=device, dtype=torch.float32)
    return torch.cat([arm_now, grip_open], dim=0)


def pick_action_from_chunk(action_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if action_tensor.ndim == 1:
        chunk = action_tensor.view(1, -1)
        return action_tensor, chunk, 0
    if action_tensor.ndim == 2:
        return action_tensor[0], action_tensor, 0
    raise ValueError(f"Unsupported action shape from server: {tuple(action_tensor.shape)}")


def _split_arm_gripper(action_1d: torch.Tensor, arm_dof: int) -> tuple[torch.Tensor, float]:
    if action_1d.numel() < arm_dof:
        raise ValueError(f"Policy action dim={action_1d.numel()} is smaller than arm dof={arm_dof}.")
    arm_target = action_1d[:arm_dof].view(1, arm_dof)
    grip_value = float(action_1d[arm_dof].item()) if action_1d.numel() > arm_dof else 1.0
    return arm_target, grip_value


def apply_franka_action_threshold(
    *,
    robot,
    action_1d: torch.Tensor,
    arm_joint_ids,
    gripper_joint_ids,
    gripper_open_pos: float,
    gripper_threshold: float,
) -> bool:
    arm_target, grip_value = _split_arm_gripper(action_1d, len(arm_joint_ids))
    grip_open = grip_value >= float(gripper_threshold)
    grip_joint_count = len(gripper_joint_ids)
    if grip_open:
        gripper_target = torch.full((1, grip_joint_count), float(gripper_open_pos), device=action_1d.device)
    else:
        gripper_target = torch.zeros((1, grip_joint_count), device=action_1d.device)

    robot.set_joint_position_target(arm_target, joint_ids=arm_joint_ids)
    robot.set_joint_position_target(gripper_target, joint_ids=gripper_joint_ids)
    robot.write_data_to_sim()
    return grip_open


def apply_franka_action_hysteresis(
    *,
    robot,
    action_1d: torch.Tensor,
    arm_joint_ids,
    gripper_joint_ids,
    gripper_open_pos: float,
    prev_gripper_open: bool,
    gripper_open_threshold: float,
    gripper_close_threshold: float,
) -> bool:
    arm_target, grip_value = _split_arm_gripper(action_1d, len(arm_joint_ids))
    if prev_gripper_open:
        grip_open = grip_value > float(gripper_close_threshold)
    else:
        grip_open = grip_value >= float(gripper_open_threshold)

    grip_joint_count = len(gripper_joint_ids)
    if grip_open:
        gripper_target = torch.full((1, grip_joint_count), float(gripper_open_pos), device=action_1d.device)
    else:
        gripper_target = torch.zeros((1, grip_joint_count), device=action_1d.device)

    robot.set_joint_position_target(arm_target, joint_ids=arm_joint_ids)
    robot.set_joint_position_target(gripper_target, joint_ids=gripper_joint_ids)
    robot.write_data_to_sim()
    return grip_open

