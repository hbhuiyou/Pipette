from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VirtualButtonLidOpenerConfig:
    button_prim_path: str = ""
    joint_prim_path: str = ""
    trigger_padding_m: float = 0.03
    open_target_position_deg: float = -40.0
    open_max_force: float = 1000.0
    open_stiffness: float = 200.0
    open_damping: float = 20.0
    default_target_position_deg: float = 0.0
    default_target_velocity_deg: float = 0.0
    default_stiffness: float = 0.0
    default_damping: float = 5.0
    default_max_force: float = 1000.0
    open_duration_seconds: float = 2.0


class VirtualButtonLidOpener:
    """Open the centrifuge lid when the robot gripper reaches a virtual button."""

    def __init__(self, config: VirtualButtonLidOpenerConfig | None = None, device: str = "cpu"):
        self.config = config if config is not None else VirtualButtonLidOpenerConfig()
        self.device = device

        self.button_prim_path = ""
        self.joint_prim_path = ""
        self._button_min_w: torch.Tensor | None = None
        self._button_max_w: torch.Tensor | None = None
        self._initialized = False
        self._has_triggered = False
        self._opening = False
        self._open_elapsed_s = 0.0
        self._open_start_position_deg = 0.0
        self._warned_unavailable = False

    @property
    def is_ready(self) -> bool:
        return (
            self._initialized
            and self._button_min_w is not None
            and self._button_max_w is not None
            and bool(self.joint_prim_path)
        )

    @property
    def has_triggered(self) -> bool:
        return self._has_triggered

    def initialize(self) -> bool:
        if self._initialized:
            return self.is_ready

        self.button_prim_path = self._require_valid_prim_path(
            self.config.button_prim_path,
            label="virtual lid button",
        )
        self.joint_prim_path = self._require_valid_prim_path(
            self.config.joint_prim_path,
            label="centrifuge lid joint",
        )

        if self.button_prim_path:
            bounds = self._get_prim_world_aligned_bounds(self.button_prim_path)
            if bounds is not None:
                min_xyz, max_xyz = bounds
                padding = max(0.0, float(self.config.trigger_padding_m))
                self._button_min_w = torch.tensor(
                    [min_xyz], dtype=torch.float32, device=self.device
                ) - padding
                self._button_max_w = torch.tensor(
                    [max_xyz], dtype=torch.float32, device=self.device
                ) + padding

        self._initialized = True
        if not self.is_ready:
            print(
                "[WARN] Virtual lid button opener is unavailable "
                f"(button='{self.button_prim_path}', joint='{self.joint_prim_path}')."
            )
        return self.is_ready

    def reset(self) -> None:
        self._has_triggered = False
        self._opening = False
        self._open_elapsed_s = 0.0
        self._open_start_position_deg = 0.0
        self._warned_unavailable = False

    def reset_to_default_closed_state(self) -> None:
        self.reset()
        self.restore_default_drive()

    def update(self, gripper_pos_w: torch.Tensor, dt_seconds: float) -> bool:
        dt_seconds = max(0.0, float(dt_seconds))
        if not self.initialize():
            if not self._warned_unavailable:
                print("[WARN] Skip virtual button lid check because opener initialization failed.")
                self._warned_unavailable = True
            return False

        if self._opening:
            self._open_elapsed_s += dt_seconds
            self._apply_open_ramp_target()
            if self._open_elapsed_s >= float(self.config.open_duration_seconds):
                self.restore_default_drive()
                self._opening = False

        if self._has_triggered:
            return False

        if gripper_pos_w is None:
            return False

        gripper_pos = gripper_pos_w[:, :3].to(device=self.device, dtype=torch.float32)
        inside_button = torch.all(
            (gripper_pos >= self._button_min_w) & (gripper_pos <= self._button_max_w),
            dim=-1,
        )
        if not bool(torch.any(inside_button).item()):
            return False

        trigger_pos = gripper_pos[inside_button][0].detach().cpu().tolist()
        self._trigger_lid_open(trigger_pos)
        return True

    def _trigger_lid_open(self, trigger_pos: list[float]) -> None:
        drive_api = self._get_lid_drive()
        if drive_api is None:
            print(f"[WARN] Cannot open lid: invalid RevoluteJoint path '{self.joint_prim_path}'.")
            return

        drive_api.CreateMaxForceAttr().Set(float(self.config.open_max_force))
        drive_api.CreateStiffnessAttr().Set(float(self.config.open_stiffness))
        drive_api.CreateDampingAttr().Set(float(self.config.open_damping))
        self._open_start_position_deg = self._read_drive_target_position(drive_api)
        drive_api.CreateTargetPositionAttr().Set(float(self._open_start_position_deg))

        self._has_triggered = True
        self._opening = True
        self._open_elapsed_s = 0.0

    def _apply_open_ramp_target(self) -> None:
        drive_api = self._get_lid_drive()
        if drive_api is None:
            return

        duration_s = max(1.0e-6, float(self.config.open_duration_seconds))
        alpha = min(1.0, self._open_elapsed_s / duration_s)
        target = self._open_start_position_deg + alpha * (
            float(self.config.open_target_position_deg) - self._open_start_position_deg
        )
        drive_api.CreateTargetPositionAttr().Set(float(target))

    def restore_default_drive(self) -> None:
        if not self.initialize():
            return
        drive_api = self._get_lid_drive()
        if drive_api is None:
            return

        drive_api.CreateMaxForceAttr().Set(float(self.config.default_max_force))
        drive_api.CreateStiffnessAttr().Set(float(self.config.default_stiffness))
        drive_api.CreateDampingAttr().Set(float(self.config.default_damping))
        drive_api.CreateTargetPositionAttr().Set(float(self.config.default_target_position_deg))
        drive_api.CreateTargetVelocityAttr().Set(float(self.config.default_target_velocity_deg))

    def get_drive_target_position(self) -> float | None:
        if not self.initialize():
            return None

        drive_api = self._get_lid_drive()
        if drive_api is None:
            return None

        return self._read_drive_target_position(drive_api)

    def _read_drive_target_position(self, drive_api) -> float:
        try:
            attr = drive_api.GetTargetPositionAttr()
            if attr and attr.HasAuthoredValueOpinion():
                value = attr.Get()
                if value is not None:
                    return float(value)
        except Exception:
            pass
        return 0.0

    def _compute_nearest_aabb_gap(self, points_w: torch.Tensor) -> float:
        if self._button_min_w is None or self._button_max_w is None:
            return float("inf")

        below = torch.clamp(self._button_min_w - points_w, min=0.0)
        above = torch.clamp(points_w - self._button_max_w, min=0.0)
        outside_delta = below + above
        gaps = torch.norm(outside_delta, dim=-1)
        return float(gaps.min().item())

    def _get_lid_drive(self):
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None or not self.joint_prim_path:
            return None

        joint_prim = stage.GetPrimAtPath(self.joint_prim_path)
        if not joint_prim.IsValid():
            return None

        return UsdPhysics.DriveAPI.Apply(joint_prim, "angular")

    @staticmethod
    def _require_valid_prim_path(configured_prim_path: str, *, label: str) -> str:
        import omni.usd

        prim_path = str(configured_prim_path).strip()
        if not prim_path:
            raise ValueError(f"Missing configured {label} prim path.")

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError(f"Cannot validate {label} prim path: USD stage is unavailable.")

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(
                f"Configured {label} prim path does not exist in the loaded USD scene: '{prim_path}'."
            )
        return prim_path

    def _get_prim_world_aligned_bounds(
        self, prim_path: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None

        try:
            purposes = [UsdGeom.Tokens.default_]
            bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
            aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if not aligned_box.IsEmpty():
                min_xyz = aligned_box.GetMin()
                max_xyz = aligned_box.GetMax()
                return (
                    (float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])),
                    (float(max_xyz[0]), float(max_xyz[1]), float(max_xyz[2])),
                )
        except Exception:
            pass

        try:
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = matrix.ExtractTranslation()
        except Exception:
            return None

        point = (float(translation[0]), float(translation[1]), float(translation[2]))
        return point, point
