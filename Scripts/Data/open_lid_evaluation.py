from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriveTargetEvalResult:
    success: bool
    reason: str
    metrics: dict[str, object]


def build_open_lid_drive_eval_result(
    *,
    initial_target_deg: float | None,
    min_target_deg: float | None,
    max_target_deg: float | None,
    triggered: bool,
    success_threshold_deg: float,
) -> DriveTargetEvalResult:
    metrics: dict[str, object] = {
        "open_lid_eval_source": "drive_target_position",
        "open_lid_virtual_button_triggered": bool(triggered),
        "open_lid_drive_target_initial_deg": float(initial_target_deg)
        if initial_target_deg is not None
        else "unavailable",
        "open_lid_drive_target_min_deg": float(min_target_deg) if min_target_deg is not None else "unavailable",
        "open_lid_drive_target_max_deg": float(max_target_deg) if max_target_deg is not None else "unavailable",
        "open_lid_drive_target_success_threshold_deg": float(success_threshold_deg),
    }

    if initial_target_deg is None or min_target_deg is None or max_target_deg is None:
        metrics["open_lid_drive_target_range_deg"] = 0.0
        metrics["open_lid_drive_target_change_abs_deg"] = 0.0
        return DriveTargetEvalResult(
            success=False,
            reason="open_lid_drive_target_unavailable",
            metrics=metrics,
        )

    target_range = float(max_target_deg) - float(min_target_deg)
    change_abs = max(
        abs(float(max_target_deg) - float(initial_target_deg)),
        abs(float(min_target_deg) - float(initial_target_deg)),
    )
    metrics["open_lid_drive_target_range_deg"] = float(target_range)
    metrics["open_lid_drive_target_change_abs_deg"] = float(change_abs)

    if not triggered:
        return DriveTargetEvalResult(
            success=False,
            reason="open_lid_virtual_button_not_triggered",
            metrics=metrics,
        )

    success = float(min_target_deg) < float(success_threshold_deg)
    return DriveTargetEvalResult(
        success=success,
        reason="success" if success else "open_lid_drive_target_threshold_not_reached",
        metrics=metrics,
    )
