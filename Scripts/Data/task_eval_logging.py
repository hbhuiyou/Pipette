from __future__ import annotations

import math
from typing import Any


def _metric_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None or isinstance(value, str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_eval_progress_value(eval_result) -> float:
    metrics = dict(getattr(eval_result, "metrics", {}) or {})
    for key in ("open_lid_drive_target_min_deg", "water_bath_lid_x_drop", "tube_z", "pipette_z", "final_z", "max_z", "final_x_deg"):
        value = _metric_float(metrics, key)
        if value is not None:
            return value
    return float("nan")


def should_print_eval_step(step_index: int, interval_steps: int, total_steps: int | None = None) -> bool:
    interval = int(interval_steps)
    if interval <= 0:
        return False
    step = int(step_index)
    if total_steps is not None and (step == 0 or step == int(total_steps) - 1):
        return True
    return step % interval == 0


def _step_label(step_index: int, total_steps: int | None) -> str:
    label = f"Step {int(step_index):03d}"
    if total_steps is not None:
        label += f"/{int(total_steps)}"
    return label


def _prefix(episode_index: int | None) -> str:
    if episode_index is None:
        return ""
    return f"[EP {int(episode_index):03d}] "


def format_task_eval_step_logs(
    *,
    task_preset,
    eval_result,
    evaluator=None,
    step_index: int,
    total_steps: int | None = None,
    episode_index: int | None = None,
    drive_target_current_deg: float | None = None,
) -> list[str]:
    task_id = str(getattr(task_preset, "task_id", "")).lower()
    metrics = dict(getattr(eval_result, "metrics", {}) or {})
    prefix = _prefix(episode_index)
    step = _step_label(step_index, total_steps)

    if task_id == "open_the_centrifuge_lid":
        initial = _metric_float(metrics, "open_lid_drive_target_initial_deg")
        min_value = _metric_float(metrics, "open_lid_drive_target_min_deg")
        max_value = _metric_float(metrics, "open_lid_drive_target_max_deg")
        current = drive_target_current_deg
        if current is None:
            current = min_value
        if current is None or initial is None:
            return [f"{prefix}[DRIVE] {step} TargetPosition=unavailable"]
        if min_value is None:
            min_value = current
        if max_value is None:
            max_value = current
        delta = float(current) - float(initial)
        return [
            f"{prefix}[DRIVE] {step} "
            f"TargetPosition={float(current):.3f} deg, "
            f"Delta={delta:+.3f} deg, "
            f"Range=[{float(min_value):.3f}, {float(max_value):.3f}] deg"
        ]

    logs: list[str] = []

    pipette_pos = getattr(evaluator, "last_pipette_pos", None) if evaluator is not None else None
    if pipette_pos is not None:
        pos = pipette_pos.detach().to("cpu")
        backend = getattr(evaluator, "_pose_backend", "") or "unknown"
        low_z_threshold = float(getattr(evaluator, "_low_z_threshold", float("nan")))
        if hasattr(evaluator, "_success_y_threshold") and hasattr(evaluator, "_success_z_threshold"):
            success_y_threshold = float(getattr(evaluator, "_success_y_threshold", float("nan")))
            success_z_threshold = float(getattr(evaluator, "_success_z_threshold", float("nan")))
            success_hold_s = float(getattr(evaluator, "_success_hold_s", float("nan")))
            required_hold_s = float(getattr(evaluator, "_required_hold_s", float("nan")))
            logs.append(
                f"{prefix}[DEBUG] {step} Pipette Pos ({backend}): "
                f"X={pos[0,0]:.4f}, Y={pos[0,1]:.4f}, Z={pos[0,2]:.4f}, "
                f"success_y>{success_y_threshold:.4f}, "
                f"success_z>{success_z_threshold:.4f}, "
                f"hold_s={success_hold_s:.3f}/{required_hold_s:.3f}, "
                f"low_z_threshold={low_z_threshold:.4f}"
            )
        else:
            distance_xy = float(getattr(evaluator, "_last_distance_xy", float("nan")))
            success_xy_distance = float(getattr(evaluator, "_success_xy_distance", float("nan")))
            distance_to_success_xy = float(getattr(evaluator, "_last_distance_to_success_xy", float("nan")))
            logs.append(
                f"{prefix}[DEBUG] {step} Pipette Pos ({backend}): "
                f"X={pos[0,0]:.4f}, Y={pos[0,1]:.4f}, Z={pos[0,2]:.4f}, "
                f"distance_xy={distance_xy:.4f}, "
                f"success_xy_distance={success_xy_distance:.4f}, "
                f"distance_to_success_xy={distance_to_success_xy:.4f}, "
                f"low_z_threshold={low_z_threshold:.4f}"
            )

    petri_close_pos = getattr(evaluator, "last_petri_close_pos", None) if evaluator is not None else None
    if petri_close_pos is not None:
        pos = petri_close_pos.detach().to("cpu")
        backend = getattr(evaluator, "_pose_backend", "") or "unknown"
        logs.append(
            f"{prefix}[DEBUG] {step} Petri_close Pos ({backend}): "
            f"X={pos[0,0]:.4f}, Y={pos[0,1]:.4f}, Z={pos[0,2]:.4f}"
        )

    tube_pos = getattr(evaluator, "last_tube_pos", None) if evaluator is not None else None
    if tube_pos is not None:
        pos = tube_pos.detach().to("cpu")
        backend = getattr(evaluator, "_pose_backend", "") or "unknown"
        if task_id == "take_the_centrifuge_tube_from_the_balance":
            roll_deg = float(getattr(evaluator, "_last_roll_deg", float("nan")))
            pitch_deg = float(getattr(evaluator, "_last_pitch_deg", float("nan")))
            yaw_deg = float(getattr(evaluator, "_last_yaw_deg", float("nan")))
            tilt_deg = float(getattr(evaluator, "_last_tilt_deg", float("nan")))
            max_tilt_deg = float(getattr(evaluator, "_max_tilt_deg", float("nan")))
            y_rise = float(getattr(evaluator, "_last_y_rise", float("nan")))
            z_drop = float(getattr(evaluator, "_last_z_drop", float("nan")))
            y_rise_threshold = float(getattr(evaluator, "_success_y_rise_threshold", float("nan")))
            z_drop_threshold = float(getattr(evaluator, "_success_z_drop_threshold", float("nan")))
            hold_s = float(getattr(evaluator, "_above_hold", float("nan")))
            required_hold_s = float(getattr(getattr(evaluator, "tube_eval", None), "success_hold_seconds", float("nan")))
            tilt_fail_deg = float(getattr(getattr(evaluator, "tube_eval", None), "tilt_fail_deg", float("nan")))
            height_delta = float(pos[0, 2]) - float(getattr(evaluator, "_initial_z", float("nan")))
            logs.append(
                f"{prefix}[DEBUG] {step} CentrifugeTube Pos ({backend}): "
                f"X={pos[0,0]:.4f}, Y={pos[0,1]:.4f}, Z={pos[0,2]:.4f}, "
                f"height_delta={height_delta:.4f}, "
                f"y_rise={y_rise:.4f}/{y_rise_threshold:.4f}, "
                f"z_drop={z_drop:.4f}/{z_drop_threshold:.4f}, "
                f"hold_s={hold_s:.3f}/{required_hold_s:.3f}, "
                f"rot_xyz_deg=({roll_deg:.2f}, {pitch_deg:.2f}, {yaw_deg:.2f}), "
                f"tilt_from_initial={tilt_deg:.2f}/{tilt_fail_deg:.2f}, max_tilt={max_tilt_deg:.2f}"
            )
            return logs

        distance_xy = float(getattr(evaluator, "_last_distance_xy", float("nan")))
        success_radius = float(getattr(evaluator, "_last_success_radius", float("nan")))
        inside_hold_s = float(getattr(evaluator, "_inside_hold_s", float("nan")))
        tube_z_min = float(getattr(evaluator, "tube_z_min", float("nan")))
        tube_z_max = float(getattr(evaluator, "tube_z_max", float("nan")))
        logs.append(
            f"{prefix}[DEBUG] {step} Tube Pos ({backend}): "
            f"X={pos[0,0]:.4f}, Y={pos[0,1]:.4f}, Z={pos[0,2]:.4f}, "
            f"distance_xy={distance_xy:.4f}, success_radius={success_radius:.4f}, "
            f"z_range=({tube_z_min:.4f}, {tube_z_max:.4f}), "
            f"inside_hold_s={inside_hold_s:.3f}"
        )

    water_bath_lid_pos = getattr(evaluator, "last_water_bath_lid_pos", None) if evaluator is not None else None
    if water_bath_lid_pos is not None:
        pos = water_bath_lid_pos.detach().to("cpu")
        backend = getattr(evaluator, "_pose_backend", "") or "unknown"
        x_drop = float(getattr(evaluator, "_last_x_drop", float("nan")))
        z_rise = float(getattr(evaluator, "_last_z_rise", float("nan")))
        success_x_drop_threshold = float(getattr(evaluator, "_success_x_drop_threshold", float("nan")))
        success_z_rise_threshold = float(getattr(evaluator, "_success_z_rise_threshold", float("nan")))
        success_hold_s = float(getattr(evaluator, "_success_hold_s", float("nan")))
        required_hold_s = float(getattr(evaluator, "_required_hold_s", float("nan")))
        lid_path = str(getattr(evaluator, "lid_prim_path", ""))
        pose_accessor = getattr(evaluator, "pose_accessor", None)
        resolved_path = str(getattr(pose_accessor, "prim_path", "")) if pose_accessor is not None else ""
        logs.append(
            f"{prefix}[DEBUG] {step} WaterBath Lid Pos ({backend}): "
            f"X={pos[0,0]:.4f}, Y={pos[0,1]:.4f}, Z={pos[0,2]:.4f}, "
            f"x_drop={x_drop:.4f}/{success_x_drop_threshold:.4f}, "
            f"z_rise={z_rise:.4f}/{success_z_rise_threshold:.4f}, "
            f"hold_s={success_hold_s:.3f}/{required_hold_s:.3f}, "
            f"lid='{lid_path}', rigid='{resolved_path}'"
        )

    if logs:
        return logs

    current_value = get_eval_progress_value(eval_result)
    if not math.isfinite(float(current_value)):
        return [f"{prefix}[EVAL] {step} current_z=nan"]

    if task_id == "place_the_centrifuge_tube_on_the_balance":
        metric_name = "tube_z"
    elif task_id == "open_the_water_bath_lid":
        metric_name = "water_bath_lid_x_drop"
    elif "centrifuge" in task_id or "spectrophotometer" in task_id:
        metric_name = "lid_x_deg"
        angle_source = str(metrics.get("angle_source") or "unknown")
        pose_backend = str(metrics.get("pose_backend") or "unknown")
        return [
            f"{prefix}[EVAL] {step} {metric_name}={float(current_value):.5f}, "
            f"source={angle_source}, backend={pose_backend}"
        ]
    else:
        metric_name = "current_z"
    return [f"{prefix}[EVAL] {step} {metric_name}={float(current_value):.5f}"]
