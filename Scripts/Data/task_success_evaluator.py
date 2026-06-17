from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _capture_prim_pose_tensor(prim_path: str, device: str) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
    if not prim_path:
        return None, None

    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None, None

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None

    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    trans = mat.ExtractTranslation()
    quat = mat.ExtractRotationQuat()

    pos = torch.tensor([[float(trans[0]), float(trans[1]), float(trans[2])]], dtype=torch.float32, device=device)
    rot = torch.tensor(
        [[float(quat.GetReal()), float(quat.GetImaginary()[0]), float(quat.GetImaginary()[1]), float(quat.GetImaginary()[2])]],
        dtype=torch.float32,
        device=device,
    )
    return pos, rot


def _capture_prim_world_xy_circle_tensor(
    prim_path: str,
    radius_scale: float,
    device: str,
) -> tuple[torch.Tensor, float] | tuple[None, None]:
    if not prim_path:
        return None, None

    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None, None

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if aligned_box.IsEmpty():
        return None, None

    min_pt = aligned_box.GetMin()
    max_pt = aligned_box.GetMax()
    center = Gf.Vec3d(
        (float(min_pt[0]) + float(max_pt[0])) * 0.5,
        (float(min_pt[1]) + float(max_pt[1])) * 0.5,
        (float(min_pt[2]) + float(max_pt[2])) * 0.5,
    )
    diameter_x = abs(float(max_pt[0]) - float(min_pt[0]))
    diameter_y = abs(float(max_pt[1]) - float(min_pt[1]))
    radius = 0.5 * min(diameter_x, diameter_y) * float(radius_scale)
    if radius <= 0.0:
        return None, None

    center_xy = torch.tensor([[float(center[0]), float(center[1])]], dtype=torch.float32, device=device)
    return center_xy, radius


def _capture_prim_local_quat_tensor(prim_path: str, device: str) -> torch.Tensor | None:
    if not prim_path:
        return None

    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None

        xform = UsdGeom.Xformable(prim)
        local_result = xform.GetLocalTransformation()
        local_mat = local_result[0] if isinstance(local_result, tuple) else local_result
        quat = local_mat.ExtractRotationQuat()
        return torch.tensor(
            [[float(quat.GetReal()), float(quat.GetImaginary()[0]), float(quat.GetImaginary()[1]), float(quat.GetImaginary()[2])]],
            dtype=torch.float32,
            device=device,
        )
    except Exception:
        return None


def _find_first_rigidbody_prim_path(prim_path: str) -> str:
    if not prim_path:
        return ""

    try:
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return ""

        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            return ""

        if root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim_path

        for child_prim in Usd.PrimRange(root_prim):
            if child_prim == root_prim:
                continue
            if child_prim.IsValid() and child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                return str(child_prim.GetPath())
    except Exception:
        return ""

    return ""


def _find_revolute_joint_body_paths(root_prim_path: str) -> tuple[str, str, str]:
    if not root_prim_path:
        return "", "", ""

    try:
        import omni.usd
        from pxr import Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return "", "", ""

        root_prim = stage.GetPrimAtPath(root_prim_path)
        if not root_prim.IsValid():
            return "", "", ""

        for prim in Usd.PrimRange(root_prim):
            if not prim.IsValid():
                continue
            type_name = prim.GetTypeName()
            is_revolute = type_name == "PhysicsRevoluteJoint"
            try:
                is_revolute = is_revolute or prim.IsA(UsdPhysics.RevoluteJoint)
            except Exception:
                pass
            if not is_revolute:
                continue

            body0_targets = prim.GetRelationship("physics:body0").GetTargets()
            body1_targets = prim.GetRelationship("physics:body1").GetTargets()
            body0_path = str(body0_targets[0]) if body0_targets else ""
            body1_path = str(body1_targets[0]) if body1_targets else ""
            return str(prim.GetPath()), body0_path, body1_path
    except Exception:
        return "", "", ""

    return "", "", ""


class _FixedPoseAccessor:
    def __init__(
        self,
        prim_path: str,
        device: str,
        *,
        view_name_prefix: str,
        backend_priority: tuple[str, ...],
        resolve_rigid_body: bool,
    ):
        self.authored_prim_path = prim_path
        self.prim_path = prim_path
        self.device = device
        self._view_name_prefix = view_name_prefix
        self._backend_priority = backend_priority
        self._rigid_object = None
        self._rigid_prim = None
        self._selected_backend = ""
        self._last_error = ""
        self._reported_errors: set[str] = set()

        if resolve_rigid_body:
            rigid_prim_path = _find_first_rigidbody_prim_path(prim_path)
            if rigid_prim_path:
                self.prim_path = rigid_prim_path
            else:
                self._select_usd_backend()
                self._announce_selected_backend()
                return

        self._select_backend()
        self._announce_selected_backend()

    @property
    def last_backend(self) -> str:
        return self._selected_backend

    @property
    def last_error(self) -> str:
        return self._last_error

    def _select_backend(self) -> None:
        initialization_errors: list[str] = []
        for backend in self._backend_priority:
            try:
                if backend == "rigid_object":
                    self._initialize_rigid_object()
                elif backend == "rigid_prim":
                    self._initialize_rigid_prim()
                elif backend == "usd":
                    self._select_usd_backend()
                else:
                    raise ValueError(f"Unsupported pose backend: {backend}")
                self._selected_backend = backend
                if initialization_errors:
                    print(
                        f"[WARN] Pose backend initialization fallback for '{self.authored_prim_path}': "
                        f"selected={backend}; unavailable={'; '.join(initialization_errors)}"
                    )
                return
            except Exception as exc:
                initialization_errors.append(f"{backend}: {exc}")

        details = "; ".join(initialization_errors) or "no backend candidates"
        raise RuntimeError(
            f"Cannot initialize a pose backend for prim '{self.authored_prim_path}'. {details}"
        )

    def _announce_selected_backend(self) -> None:
        print(
            f"[INFO] Fixed pose backend: prim='{self.authored_prim_path}', "
            f"tracked_prim='{self.prim_path}', backend={self._selected_backend}"
        )

    def _record_error(self, message: str) -> None:
        self._last_error = message
        if message not in self._reported_errors:
            self._reported_errors.add(message)
            print(
                f"[ERROR] Fixed pose backend '{self._selected_backend}' failed for "
                f"prim '{self.authored_prim_path}': {message}"
            )

    def _initialize_rigid_object(self) -> None:
        from isaaclab.assets import RigidObject, RigidObjectCfg

        cfg = RigidObjectCfg(prim_path=self.prim_path, spawn=None)
        self._rigid_object = RigidObject(cfg)

    def _initialize_rigid_prim(self) -> None:
        from isaacsim.core.prims import RigidPrim

        view_name = f"{self._view_name_prefix}_{id(self)}"
        self._rigid_prim = RigidPrim(prim_paths_expr=self.prim_path, name=view_name)
        if hasattr(self._rigid_prim, "initialize"):
            self._rigid_prim.initialize()

    def _select_usd_backend(self) -> None:
        pos, quat = _capture_prim_pose_tensor(self.authored_prim_path, self.device)
        if pos is None or quat is None:
            raise RuntimeError(
                f"USD pose is unavailable for prim '{self.authored_prim_path}'."
            )
        self._selected_backend = "usd"

    def _capture_from_rigid_object(
        self,
        dt_seconds: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if self._rigid_object is None:
            return None, None

        update_dt = float(dt_seconds) if dt_seconds is not None else 0.0
        self._rigid_object.update(update_dt)
        if self._rigid_object.data.root_pos_w is None or self._rigid_object.data.root_quat_w is None:
            return None, None

        pos = self._rigid_object.data.root_pos_w.clone().to(device=self.device, dtype=torch.float32)
        quat = self._rigid_object.data.root_quat_w.clone().to(device=self.device, dtype=torch.float32)
        return pos, quat

    def _capture_from_rigid_prim(self) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if self._rigid_prim is None:
            return None, None

        try:
            pos_raw, quat_raw = self._rigid_prim.get_world_poses(usd=False)
        except TypeError:
            pos_raw, quat_raw = self._rigid_prim.get_world_poses()

        if pos_raw is None or quat_raw is None:
            return None, None

        pos = torch.as_tensor(pos_raw, dtype=torch.float32, device=self.device)
        quat = torch.as_tensor(quat_raw, dtype=torch.float32, device=self.device)
        if pos.ndim == 1:
            pos = pos.view(1, -1)
        if quat.ndim == 1:
            quat = quat.view(1, -1)
        if pos.shape[-1] < 3 or quat.shape[-1] < 4:
            return None, None
        return pos[:, :3], quat[:, :4]

    def get_pose(
        self,
        dt_seconds: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        try:
            if self._selected_backend == "rigid_object":
                pos, quat = self._capture_from_rigid_object(dt_seconds)
            elif self._selected_backend == "rigid_prim":
                pos, quat = self._capture_from_rigid_prim()
            elif self._selected_backend == "usd":
                pos, quat = _capture_prim_pose_tensor(self.authored_prim_path, self.device)
            else:
                raise RuntimeError("Pose backend was not selected.")
        except Exception as exc:
            self._record_error(str(exc))
            return None, None

        if pos is None or quat is None:
            self._record_error(
                f"Selected pose backend '{self._selected_backend}' returned no pose "
                f"for prim '{self.authored_prim_path}'."
            )
            return None, None

        self._last_error = ""
        return pos, quat


class TubePoseAccessor(_FixedPoseAccessor):
    def __init__(self, prim_path: str, device: str):
        super().__init__(
            prim_path,
            device,
            view_name_prefix="tube_rigid_prim_view",
            backend_priority=("rigid_object", "rigid_prim", "usd"),
            resolve_rigid_body=False,
        )


class PipettePoseAccessor(_FixedPoseAccessor):
    def __init__(self, prim_path: str, device: str):
        super().__init__(
            prim_path,
            device,
            view_name_prefix="pipette_rigid_prim_view",
            backend_priority=("rigid_object", "rigid_prim", "usd"),
            resolve_rigid_body=True,
        )


class PetriDishPoseAccessor(_FixedPoseAccessor):
    def __init__(self, prim_path: str, device: str):
        super().__init__(
            prim_path,
            device,
            view_name_prefix="petri_dish_rigid_prim_view",
            backend_priority=("rigid_object", "rigid_prim", "usd"),
            resolve_rigid_body=True,
        )


class CentrifugeLidPoseAccessor(_FixedPoseAccessor):
    def __init__(self, prim_path: str, device: str):
        super().__init__(
            prim_path,
            device,
            view_name_prefix="centrifuge_lid_rigid_prim_view",
            backend_priority=("rigid_prim", "rigid_object", "usd"),
            resolve_rigid_body=True,
        )


def _quat_rotate_vector_wxyz(q_wxyz: torch.Tensor, v_xyz: torch.Tensor) -> torch.Tensor:
    q = q_wxyz / torch.norm(q_wxyz, dim=1, keepdim=True).clamp_min(1.0e-6)
    q_xyz = q[:, 1:4]
    q_w = q[:, 0:1]
    t = 2.0 * torch.cross(q_xyz, v_xyz, dim=1)
    return v_xyz + q_w * t + torch.cross(q_xyz, t, dim=1)


def _quat_conjugate_wxyz(q_wxyz: torch.Tensor) -> torch.Tensor:
    q = q_wxyz.clone()
    q[:, 1:4] = -q[:, 1:4]
    return q


def _quat_multiply_wxyz(q1_wxyz: torch.Tensor, q2_wxyz: torch.Tensor) -> torch.Tensor:
    q1 = q1_wxyz / torch.norm(q1_wxyz, dim=1, keepdim=True).clamp_min(1.0e-6)
    q2 = q2_wxyz / torch.norm(q2_wxyz, dim=1, keepdim=True).clamp_min(1.0e-6)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=1,
    )


def _relative_quat_wxyz(parent_quat_wxyz: torch.Tensor, child_quat_wxyz: torch.Tensor) -> torch.Tensor:
    return _quat_multiply_wxyz(_quat_conjugate_wxyz(parent_quat_wxyz), child_quat_wxyz)


def _compute_tilt_deg_from_initial(
    q_wxyz: torch.Tensor,
    local_up_axis: torch.Tensor,
    initial_up_world: torch.Tensor,
) -> float:
    local_up = local_up_axis.view(1, 3).to(device=q_wxyz.device, dtype=torch.float32)
    world_up = _quat_rotate_vector_wxyz(q_wxyz, local_up)
    world_up = world_up / torch.norm(world_up, dim=1, keepdim=True).clamp_min(1.0e-6)

    init_up = initial_up_world.view(1, 3).to(device=q_wxyz.device, dtype=torch.float32)
    init_up = init_up / torch.norm(init_up, dim=1, keepdim=True).clamp_min(1.0e-6)
    cos_theta = torch.sum(world_up * init_up, dim=1).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos_theta))[0].item())


def _quat_delta_angle_deg(q0: torch.Tensor, q1: torch.Tensor) -> float:
    qa = q0 / torch.norm(q0, dim=1, keepdim=True).clamp_min(1.0e-6)
    qb = q1 / torch.norm(q1, dim=1, keepdim=True).clamp_min(1.0e-6)
    dot = torch.sum(qa * qb, dim=1).abs().clamp(0.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    return float(torch.rad2deg(angle)[0].item())


def _quat_to_xyz_roll_deg(q_wxyz: torch.Tensor) -> float:
    q = q_wxyz / torch.norm(q_wxyz, dim=1, keepdim=True).clamp_min(1.0e-6)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    return float(torch.rad2deg(roll)[0].item())


def _quat_to_xyz_euler_deg(q_wxyz: torch.Tensor) -> tuple[float, float, float]:
    q = q_wxyz / torch.norm(q_wxyz, dim=1, keepdim=True).clamp_min(1.0e-6)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return (
        float(torch.rad2deg(roll)[0].item()),
        float(torch.rad2deg(pitch)[0].item()),
        float(torch.rad2deg(yaw)[0].item()),
    )


def _require_valid_prim_path(configured_prim_path: str, *, task_id: str, label: str) -> str:
    prim_path = str(configured_prim_path).strip()
    if not prim_path:
        raise ValueError(
            f"Task '{task_id}' does not define {label} prim path in task_registry."
        )

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError(
            f"Cannot validate {label} prim path for task '{task_id}': USD stage is unavailable."
        )

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(
            f"Task '{task_id}' configured invalid {label} prim path: '{prim_path}'. "
            "Update task_registry to match the loaded USD scene."
        )
    return prim_path


@dataclass
class EvalResult:
    success: bool
    reason: str
    metrics: dict[str, float | str]


class BaseTaskEvaluator:
    name: str = "noop"

    def reset_episode(self):
        raise NotImplementedError

    def update(self, dt_seconds: float):
        raise NotImplementedError

    def finalize(self) -> EvalResult:
        raise NotImplementedError


class NoopEvaluator(BaseTaskEvaluator):
    name = "noop"

    def reset_episode(self):
        return

    def update(self, dt_seconds: float):
        return

    def finalize(self) -> EvalResult:
        return EvalResult(success=True, reason="unsupported_task_skip_filter", metrics={})


class TubeTaskEvaluator(BaseTaskEvaluator):
    name = "tube"

    def __init__(self, task_preset, device: str):
        self.task_preset = task_preset
        self.device = device
        self.tube_eval = task_preset.tube_eval
        self.task_id = str(getattr(task_preset, "task_id", ""))
        self.tube_prim_path = task_preset.tube_prim_path
        self.local_up_axis = torch.tensor(task_preset.tube_local_up_axis, dtype=torch.float32, device=device)
        self.pose_accessor = TubePoseAccessor(prim_path=self.tube_prim_path, device=device)

        self._use_y_rise_z_drop_eval = (
            self.tube_eval.success_y_rise_threshold is not None
            and self.tube_eval.success_z_drop_threshold is not None
        )
        self._success_y_rise_threshold = float(self.tube_eval.success_y_rise_threshold or 0.0)
        self._success_z_drop_threshold = float(self.tube_eval.success_z_drop_threshold or 0.0)
        self._initial_y = 0.0
        self._initial_z = 0.0
        self._success_z = 0.0
        self._initial_up_world = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        self._above_hold = 0.0
        self._max_z = -math.inf
        self._max_tilt_deg = 0.0
        self._last_z = 0.0
        self._last_tube_pos = None
        self._last_tube_quat = None
        self._last_y = 0.0
        self._last_y_rise = 0.0
        self._last_z_drop = 0.0
        self._last_roll_deg = 0.0
        self._last_pitch_deg = 0.0
        self._last_yaw_deg = 0.0
        self._last_tilt_deg = 0.0
        self._pose_valid = True
        self._tilt_failed = False
        self._elapsed_s = 0.0
        self._pose_backend = ""

    @property
    def last_tube_pos(self):
        return self._last_tube_pos

    @property
    def last_tube_quat(self):
        return self._last_tube_quat

    def reset_episode(self):
        tube_pos0, tube_quat0 = self.pose_accessor.get_pose(dt_seconds=0.0)
        if tube_pos0 is None or tube_quat0 is None:
            self._pose_valid = False
            return

        self._pose_valid = True
        self._tilt_failed = False
        self._above_hold = 0.0
        self._max_tilt_deg = 0.0
        self._last_tilt_deg = 0.0
        self._elapsed_s = 0.0

        if self.tube_eval.initial_z is not None:
            self._initial_z = float(self.tube_eval.initial_z)
        else:
            self._initial_z = float(tube_pos0[0, 2].item())
        self._initial_y = float(tube_pos0[0, 1].item())

        if self.tube_eval.initial_rotation_wxyz is not None:
            initial_quat = torch.tensor([list(self.tube_eval.initial_rotation_wxyz)], dtype=torch.float32, device=self.device)
        else:
            initial_quat = tube_quat0.clone()

        self._initial_up_world = _quat_rotate_vector_wxyz(initial_quat, self.local_up_axis.view(1, 3))[0]
        self._success_z = self._initial_z + float(self.tube_eval.success_height_delta)
        self._max_z = self._initial_z
        self._last_z = self._initial_z
        self._last_y = self._initial_y
        self._last_y_rise = 0.0
        self._last_z_drop = 0.0
        self._last_tube_pos = tube_pos0.detach().clone()
        self._last_tube_quat = tube_quat0.detach().clone()
        self._last_roll_deg, self._last_pitch_deg, self._last_yaw_deg = _quat_to_xyz_euler_deg(tube_quat0)
        self._pose_backend = self.pose_accessor.last_backend

    def update(self, dt_seconds: float):
        if not self._pose_valid:
            return

        tube_pos, tube_quat = self.pose_accessor.get_pose(dt_seconds=float(dt_seconds))
        if tube_pos is None or tube_quat is None:
            self._pose_valid = False
            return

        self._pose_backend = self.pose_accessor.last_backend

        current_z = float(tube_pos[0, 2].item())
        current_y = float(tube_pos[0, 1].item())
        self._elapsed_s += float(dt_seconds)
        self._last_tube_pos = tube_pos.detach().clone()
        self._last_tube_quat = tube_quat.detach().clone()
        self._last_y = current_y
        self._last_z = current_z
        self._last_y_rise = current_y - self._initial_y
        self._last_z_drop = self._initial_z - current_z
        if current_z > self._max_z:
            self._max_z = current_z

        tilt_deg = _compute_tilt_deg_from_initial(tube_quat, self.local_up_axis, self._initial_up_world)
        self._last_tilt_deg = tilt_deg
        self._last_roll_deg, self._last_pitch_deg, self._last_yaw_deg = _quat_to_xyz_euler_deg(tube_quat)
        if tilt_deg > self._max_tilt_deg:
            self._max_tilt_deg = tilt_deg
        if tilt_deg > float(self.tube_eval.tilt_fail_deg):
            self._tilt_failed = True

        if self._use_y_rise_z_drop_eval:
            success_region = (
                self._last_y_rise > self._success_y_rise_threshold
                and self._last_z_drop > self._success_z_drop_threshold
            )
        else:
            success_region = current_z > self._success_z

        if success_region:
            self._above_hold += float(dt_seconds)
        else:
            self._above_hold = 0.0

    def finalize(self) -> EvalResult:
        if not self._pose_valid:
            return EvalResult(
                success=False,
                reason="tube_pose_unavailable",
                metrics={
                    "initial_y": self._initial_y,
                    "initial_z": self._initial_z,
                    "final_y": self._last_y,
                    "final_z": self._last_z,
                    "y_rise": self._last_y_rise,
                    "z_drop": self._last_z_drop,
                    "success_y_rise_threshold": self._success_y_rise_threshold,
                    "success_z_drop_threshold": self._success_z_drop_threshold,
                    "max_z": self._max_z,
                    "max_tilt_deg": self._max_tilt_deg,
                    "final_tilt_deg": self._last_tilt_deg,
                    "final_roll_deg": self._last_roll_deg,
                    "final_pitch_deg": self._last_pitch_deg,
                    "final_yaw_deg": self._last_yaw_deg,
                    "elapsed_s": self._elapsed_s,
                    "pose_backend": self._pose_backend or "unavailable",
                },
            )

        if self._tilt_failed:
            return EvalResult(
                success=False,
                reason="tube_tilt_fail",
                metrics={
                    "initial_y": self._initial_y,
                    "initial_z": self._initial_z,
                    "success_z": self._success_z,
                    "max_z": self._max_z,
                    "final_y": self._last_y,
                    "final_z": self._last_z,
                    "y_rise": self._last_y_rise,
                    "z_drop": self._last_z_drop,
                    "success_y_rise_threshold": self._success_y_rise_threshold,
                    "success_z_drop_threshold": self._success_z_drop_threshold,
                    "max_tilt_deg": self._max_tilt_deg,
                    "final_tilt_deg": self._last_tilt_deg,
                    "final_roll_deg": self._last_roll_deg,
                    "final_pitch_deg": self._last_pitch_deg,
                    "final_yaw_deg": self._last_yaw_deg,
                    "above_hold_s": self._above_hold,
                    "required_hold_s": float(self.tube_eval.success_hold_seconds),
                    "elapsed_s": self._elapsed_s,
                    "pose_backend": self._pose_backend or "unknown",
                },
            )

        success = self._above_hold >= float(self.tube_eval.success_hold_seconds)
        failure_reason = "tube_y_rise_z_drop_hold_not_reached" if self._use_y_rise_z_drop_eval else "tube_height_hold_not_reached"
        return EvalResult(
            success=bool(success),
            reason="success" if success else failure_reason,
            metrics={
                "initial_y": self._initial_y,
                "initial_z": self._initial_z,
                "success_z": self._success_z,
                "max_z": self._max_z,
                "final_y": self._last_y,
                "final_z": self._last_z,
                "y_rise": self._last_y_rise,
                "z_drop": self._last_z_drop,
                "success_y_rise_threshold": self._success_y_rise_threshold,
                "success_z_drop_threshold": self._success_z_drop_threshold,
                "max_tilt_deg": self._max_tilt_deg,
                "final_tilt_deg": self._last_tilt_deg,
                "final_roll_deg": self._last_roll_deg,
                "final_pitch_deg": self._last_pitch_deg,
                "final_yaw_deg": self._last_yaw_deg,
                "above_hold_s": self._above_hold,
                "required_hold_s": float(self.tube_eval.success_hold_seconds),
                "height_delta": self._last_z - self._initial_z,
                "height_delta_max": self._max_z - self._initial_z,
                "elapsed_s": self._elapsed_s,
                "pose_backend": self._pose_backend or "unknown",
                "tube_x": float(self._last_tube_pos[0, 0].detach().cpu().item()) if self._last_tube_pos is not None else 0.0,
                "tube_y": float(self._last_tube_pos[0, 1].detach().cpu().item()) if self._last_tube_pos is not None else 0.0,
                "tube_z": float(self._last_tube_pos[0, 2].detach().cpu().item()) if self._last_tube_pos is not None else self._last_z,
            },
        )


class CentrifugeTaskEvaluator(BaseTaskEvaluator):
    name = "lid_angle"

    def __init__(self, task_preset, device: str):
        self.task_preset = task_preset
        self.task_id = str(task_preset.task_id)
        self.device = device
        self.centrifuge_eval = task_preset.centrifuge_eval
        self._lid_label = "spectrophotometer_lid" if "spectrophotometer" in self.task_id.lower() else "centrifuge_lid"
        configured_lid_path = str(self.centrifuge_eval.lid_prim_path).strip()
        self.lid_prim_path = _require_valid_prim_path(
            configured_lid_path,
            task_id=self.task_id,
            label=self._lid_label,
        )
        if self.centrifuge_eval.success_x_threshold_deg is None or not self.centrifuge_eval.success_direction:
            raise ValueError(
                f"Task '{self.task_id}' uses lid-angle evaluator but does not define "
                "success_x_threshold_deg/success_direction in task_registry."
            )
        self.success_x_threshold_deg = float(self.centrifuge_eval.success_x_threshold_deg)
        self.success_direction = str(self.centrifuge_eval.success_direction).lower().strip()
        self.required_hold_s = float(self.centrifuge_eval.success_hold_seconds)
        if self.success_direction not in ("greater", "less"):
            raise ValueError(
                f"Unsupported lid success_direction='{self.centrifuge_eval.success_direction}'. "
                "Use 'greater' or 'less'."
            )
        self.joint_prim_path, joint_body0_path, joint_body1_path = _find_revolute_joint_body_paths(self.lid_prim_path)
        joint_body0_rigid_path = _find_first_rigidbody_prim_path(joint_body0_path) or joint_body0_path
        joint_body1_rigid_path = _find_first_rigidbody_prim_path(joint_body1_path) or joint_body1_path
        if joint_body1_rigid_path and (
            joint_body1_rigid_path == self.lid_prim_path or joint_body1_rigid_path.startswith(f"{self.lid_prim_path}/")
        ):
            self.lid_pose_path = joint_body1_rigid_path
            self.base_pose_path = joint_body0_rigid_path
        elif joint_body0_rigid_path and (
            joint_body0_rigid_path == self.lid_prim_path or joint_body0_rigid_path.startswith(f"{self.lid_prim_path}/")
        ):
            self.lid_pose_path = joint_body0_rigid_path
            self.base_pose_path = joint_body1_rigid_path
        else:
            self.lid_pose_path = joint_body1_rigid_path or joint_body0_rigid_path or self.lid_prim_path
            self.base_pose_path = joint_body0_rigid_path if self.lid_pose_path != joint_body0_rigid_path else joint_body1_rigid_path

        self.pose_accessor = CentrifugeLidPoseAccessor(prim_path=self.lid_prim_path, device=device)
        self.lid_link_pose_accessor = CentrifugeLidPoseAccessor(prim_path=self.lid_pose_path, device=device)
        self.base_link_pose_accessor = (
            CentrifugeLidPoseAccessor(prim_path=self.base_pose_path, device=device) if self.base_pose_path else None
        )

        self._pose_valid = True
        self._success = False
        self._above_hold_s = 0.0
        self._initial_x_deg = 0.0
        self._last_x_deg = 0.0
        self._last_world_x_deg = 0.0
        self._usd_reference_x_deg = 0.0
        self._world_to_usd_x_offset_deg = 0.0
        self._min_x_deg = math.inf
        self._max_x_deg = -math.inf
        self._elapsed_s = 0.0
        self._pose_backend = ""
        self._base_pose_backend = ""
        self._angle_source = "lid_world_pose"

    def _get_lid_angle_quat(self, dt_seconds: float) -> tuple[torch.Tensor | None, str]:
        lid_pos, lid_quat = self.lid_link_pose_accessor.get_pose(dt_seconds=dt_seconds)
        self._pose_backend = self.lid_link_pose_accessor.last_backend
        if lid_pos is None or lid_quat is None:
            return None, "unavailable"

        if self.base_link_pose_accessor is None:
            self._base_pose_backend = ""
            return lid_quat, "lid_world_pose"

        base_pos, base_quat = self.base_link_pose_accessor.get_pose(dt_seconds=dt_seconds)
        self._base_pose_backend = self.base_link_pose_accessor.last_backend
        if base_pos is None or base_quat is None:
            return lid_quat, "lid_world_pose_base_unavailable"

        return _relative_quat_wxyz(base_quat, lid_quat), "joint_relative_pose"

    def reset_episode(self):
        self._pose_valid = True
        self._success = False
        self._above_hold_s = 0.0
        self._initial_x_deg = 0.0
        self._last_x_deg = 0.0
        self._last_world_x_deg = 0.0
        self._usd_reference_x_deg = 0.0
        self._world_to_usd_x_offset_deg = 0.0
        self._min_x_deg = math.inf
        self._max_x_deg = -math.inf
        self._elapsed_s = 0.0
        self._pose_backend = ""
        self._base_pose_backend = ""
        self._angle_source = "lid_world_pose"

        angle_quat0, angle_source = self._get_lid_angle_quat(dt_seconds=0.0)
        self._angle_source = angle_source
        if angle_quat0 is None:
            self._pose_valid = False
            return

        world_x_deg = _quat_to_xyz_roll_deg(angle_quat0)
        local_quat0 = _capture_prim_local_quat_tensor(self.lid_prim_path, self.device)
        if local_quat0 is None:
            local_x_deg = world_x_deg
        else:
            local_x_deg = _quat_to_xyz_roll_deg(local_quat0)

        self._last_world_x_deg = world_x_deg
        self._usd_reference_x_deg = local_x_deg
        self._world_to_usd_x_offset_deg = local_x_deg - world_x_deg
        self._initial_x_deg = local_x_deg
        self._last_x_deg = self._initial_x_deg
        self._min_x_deg = self._initial_x_deg
        self._max_x_deg = self._initial_x_deg

    def update(self, dt_seconds: float):
        if self._success or not self._pose_valid:
            return

        angle_quat, angle_source = self._get_lid_angle_quat(dt_seconds=float(dt_seconds))
        self._angle_source = angle_source
        if angle_quat is None:
            self._pose_valid = False
            return

        self._elapsed_s += float(dt_seconds)
        current_world_x_deg = _quat_to_xyz_roll_deg(angle_quat)
        self._last_world_x_deg = current_world_x_deg
        current_x_deg = current_world_x_deg + self._world_to_usd_x_offset_deg
        self._last_x_deg = current_x_deg
        if current_x_deg < self._min_x_deg:
            self._min_x_deg = current_x_deg
        if current_x_deg > self._max_x_deg:
            self._max_x_deg = current_x_deg

        if self.success_direction == "greater":
            threshold_met = current_x_deg > self.success_x_threshold_deg
        else:
            threshold_met = current_x_deg < self.success_x_threshold_deg

        if threshold_met:
            self._above_hold_s += float(dt_seconds)
        else:
            self._above_hold_s = 0.0

        if threshold_met and self._above_hold_s >= self.required_hold_s:
            self._success = True

    def finalize(self) -> EvalResult:
        metrics = {
            "lid_prim_path": self.lid_prim_path,
            "joint_prim_path": self.joint_prim_path,
            "base_pose_path": self.base_pose_path,
            "lid_pose_path": self.lid_pose_path,
            "tracked_prim_path": self.lid_link_pose_accessor.prim_path,
            "initial_x_deg": self._initial_x_deg,
            "final_x_deg": self._last_x_deg,
            "raw_world_x_deg": self._last_world_x_deg,
            "usd_reference_x_deg": self._usd_reference_x_deg,
            "world_to_usd_x_offset_deg": self._world_to_usd_x_offset_deg,
            "min_x_deg": self._min_x_deg,
            "max_x_deg": self._max_x_deg,
            "success_x_threshold_deg": self.success_x_threshold_deg,
            "success_direction": self.success_direction,
            "above_hold_s": self._above_hold_s,
            "required_hold_s": self.required_hold_s,
            "elapsed_s": self._elapsed_s,
            "pose_backend": self._pose_backend or "unknown",
            "base_pose_backend": self._base_pose_backend or "none",
            "angle_source": self._angle_source,
        }

        if not self._pose_valid:
            return EvalResult(
                success=False,
                reason=f"{self._lid_label}_pose_unavailable",
                metrics=metrics,
            )

        if self._success:
            return EvalResult(success=True, reason="success", metrics=metrics)

        return EvalResult(success=False, reason=f"{self._lid_label}_x_threshold_not_reached", metrics=metrics)


class WaterBathLidTaskEvaluator(BaseTaskEvaluator):
    name = "water_bath_lid_position"

    def __init__(self, task_preset, device: str):
        self.task_preset = task_preset
        self.task_id = str(task_preset.task_id)
        self.device = device
        self.config = task_preset.water_bath_lid_eval
        configured_lid_path = str(self.config.lid_prim_path).strip()
        self.lid_prim_path = _require_valid_prim_path(
            configured_lid_path,
            task_id=self.task_id,
            label="water_bath_lid",
        )
        self.pose_accessor = PipettePoseAccessor(prim_path=self.lid_prim_path, device=device)

        self._success_x_drop_threshold = float(self.config.success_x_drop_threshold)
        self._success_z_rise_threshold = float(self.config.success_z_rise_threshold)
        self._required_hold_s = float(self.config.success_hold_seconds)
        self._timeout_seconds = float(self.config.timeout_seconds)
        self._elapsed_s = 0.0
        self._success_hold_s = 0.0
        self._success = False
        self._pose_valid = True
        self._pose_backend = ""
        self._last_lid_pos = None
        self._initial_x = 0.0
        self._initial_z = 0.0
        self._last_x = 0.0
        self._last_y = 0.0
        self._last_z = 0.0
        self._last_x_drop = 0.0
        self._last_z_rise = 0.0

    @property
    def last_water_bath_lid_pos(self):
        return self._last_lid_pos

    def reset_episode(self):
        self._elapsed_s = 0.0
        self._success_hold_s = 0.0
        self._success = False
        self._pose_valid = True
        self._pose_backend = ""
        self._last_lid_pos = None
        self._initial_x = 0.0
        self._initial_z = 0.0
        self._last_x = 0.0
        self._last_y = 0.0
        self._last_z = 0.0
        self._last_x_drop = 0.0
        self._last_z_rise = 0.0

        lid_pos, _ = self.pose_accessor.get_pose(dt_seconds=0.0)
        self._pose_backend = self.pose_accessor.last_backend
        if lid_pos is None:
            self._pose_valid = False
            return

        self._last_lid_pos = lid_pos.detach().clone()
        self._initial_x = float(lid_pos[0, 0].item())
        self._initial_z = float(lid_pos[0, 2].item())
        self._last_x = self._initial_x
        self._last_y = float(lid_pos[0, 1].item())
        self._last_z = float(lid_pos[0, 2].item())

    def update(self, dt_seconds: float):
        if self._success or not self._pose_valid:
            return

        dt = float(dt_seconds)
        self._elapsed_s += dt
        lid_pos, _ = self.pose_accessor.get_pose(dt_seconds=dt)
        self._pose_backend = self.pose_accessor.last_backend
        if lid_pos is None:
            self._pose_valid = False
            self._success_hold_s = 0.0
            return

        self._last_lid_pos = lid_pos.detach().clone()
        self._last_x = float(lid_pos[0, 0].item())
        self._last_y = float(lid_pos[0, 1].item())
        self._last_z = float(lid_pos[0, 2].item())
        self._last_x_drop = self._initial_x - self._last_x
        self._last_z_rise = self._last_z - self._initial_z
        in_success_region = (
            self._last_x_drop > self._success_x_drop_threshold
            and self._last_z_rise > self._success_z_rise_threshold
        )
        if in_success_region:
            self._success_hold_s += dt
        else:
            self._success_hold_s = 0.0

        if self._success_hold_s >= self._required_hold_s:
            self._success = True

    def _metrics(self) -> dict[str, float | str | bool]:
        return {
            "lid_prim_path": self.lid_prim_path,
            "resolved_rigid_prim_path": getattr(self.pose_accessor, "prim_path", ""),
            "elapsed_s": self._elapsed_s,
            "pose_backend": self._pose_backend or "unknown",
            "initial_x": self._initial_x,
            "initial_z": self._initial_z,
            "water_bath_lid_x": self._last_x,
            "water_bath_lid_y": self._last_y,
            "water_bath_lid_z": self._last_z,
            "water_bath_lid_x_drop": self._last_x_drop,
            "water_bath_lid_z_rise": self._last_z_rise,
            "final_x": self._last_x,
            "final_y": self._last_y,
            "final_z": self._last_z,
            "success_x_drop_threshold": self._success_x_drop_threshold,
            "success_z_rise_threshold": self._success_z_rise_threshold,
            "success_hold_s": self._success_hold_s,
            "required_hold_s": self._required_hold_s,
            "in_success_region": bool(
                self._last_x_drop > self._success_x_drop_threshold
                and self._last_z_rise > self._success_z_rise_threshold
            ),
        }

    def finalize(self) -> EvalResult:
        metrics = self._metrics()
        if not self._pose_valid:
            return EvalResult(success=False, reason="water_bath_lid_pose_unavailable", metrics=metrics)
        if self._success:
            return EvalResult(success=True, reason="success", metrics=metrics)
        return EvalResult(success=False, reason="water_bath_lid_delta_hold_not_reached", metrics=metrics)


class PipetteTaskEvaluator(BaseTaskEvaluator):
    name = "pipette"

    def __init__(self, task_preset, device: str):
        self.task_preset = task_preset
        self.device = device
        self.pipette_prim_path = task_preset.pipette_prim_path
        self.petri_prim_path = task_preset.petri_prim_path
        self.pipette_eval = task_preset.pipette_eval
        self.pipette_pose_accessor = PipettePoseAccessor(prim_path=self.pipette_prim_path, device=device)
        self._elapsed_s = 0.0
        self._success = False
        self._failed_low_z = False
        self._pose_backend = ""
        self._last_pipette_pos = None
        self._last_z = 0.0
        self._min_z = math.inf
        self._success_xy_distance = float(self.pipette_eval.success_xy_distance)
        self._low_z_threshold = float(self.pipette_eval.low_z_threshold)
        self._last_distance_xy = math.inf
        self._last_distance_to_success_xy = math.inf

    def reset_episode(self):
        self._elapsed_s = 0.0
        self._success = False
        self._failed_low_z = False
        self._pose_backend = ""
        self._last_pipette_pos = None
        self._last_z = 0.0
        self._min_z = math.inf
        self._last_distance_xy = math.inf
        self._last_distance_to_success_xy = math.inf

    @property
    def last_pipette_pos(self):
        return self._last_pipette_pos

    def update(self, dt_seconds: float):
        if self._failed_low_z:
            return

        self._elapsed_s += float(dt_seconds)

        pipette_pos, _ = self.pipette_pose_accessor.get_pose(dt_seconds=float(dt_seconds))
        self._pose_backend = self.pipette_pose_accessor.last_backend
        self._last_pipette_pos = pipette_pos.detach().clone() if pipette_pos is not None else None
        petri_pos, _ = _capture_prim_pose_tensor(self.petri_prim_path, self.device)

        if pipette_pos is not None and petri_pos is not None:
            current_z = float(pipette_pos[0, 2].item())
            self._last_z = current_z
            if current_z < self._min_z:
                self._min_z = current_z
            if current_z < self._low_z_threshold:
                self._failed_low_z = True
                return

            dist = torch.norm(pipette_pos[0, :2] - petri_pos[0, :2])
            self._last_distance_xy = float(dist.item())
            self._last_distance_to_success_xy = max(0.0, self._last_distance_xy - self._success_xy_distance)
            if self._last_distance_xy <= self._success_xy_distance:
                self._success = True

    def finalize(self) -> EvalResult:
        if self._failed_low_z:
            return EvalResult(
                success=False,
                reason="pipette_z_below_threshold",
                metrics={
                    "elapsed_s": self._elapsed_s,
                    "pose_backend": self._pose_backend or "unknown",
                    "final_z": self._last_z,
                    "min_z": self._min_z,
                    "low_z_threshold": self._low_z_threshold,
                    "success_xy_distance": self._success_xy_distance,
                    "distance_xy": self._last_distance_xy,
                    "distance_to_success_xy": self._last_distance_to_success_xy,
                },
            )

        if self._success:
            return EvalResult(
                success=True,
                reason="success",
                metrics={
                    "elapsed_s": self._elapsed_s,
                    "pose_backend": self._pose_backend or "unknown",
                    "final_z": self._last_z,
                    "min_z": self._min_z,
                    "low_z_threshold": self._low_z_threshold,
                    "success_xy_distance": self._success_xy_distance,
                    "distance_xy": self._last_distance_xy,
                    "distance_to_success_xy": self._last_distance_to_success_xy,
                },
            )
        else:
            return EvalResult(
                success=False,
                reason="timeout",
                metrics={
                    "elapsed_s": self._elapsed_s,
                    "pose_backend": self._pose_backend or "unknown",
                    "final_z": self._last_z,
                    "min_z": self._min_z,
                    "low_z_threshold": self._low_z_threshold,
                    "success_xy_distance": self._success_xy_distance,
                    "distance_xy": self._last_distance_xy,
                    "distance_to_success_xy": self._last_distance_to_success_xy,
                },
            )


class PlacePipetteOnStandTaskEvaluator(BaseTaskEvaluator):
    name = "place_pipette_on_stand"

    def __init__(self, task_preset, device: str):
        self.task_preset = task_preset
        self.device = device
        self.pipette_prim_path = task_preset.pipette_prim_path
        self.pipette_eval = task_preset.pipette_eval
        self.config = task_preset.place_pipette_on_stand_eval
        self.pipette_pose_accessor = PipettePoseAccessor(prim_path=self.pipette_prim_path, device=device)

        self._elapsed_s = 0.0
        self._success = False
        self._failed_low_z = False
        self._pose_backend = ""
        self._last_pipette_pos = None
        self._last_y = 0.0
        self._last_z = 0.0
        self._min_z = math.inf
        self._success_hold_s = 0.0
        self._success_y_threshold = float(self.config.success_y_threshold)
        self._success_z_threshold = float(self.config.success_z_threshold)
        self._required_hold_s = float(self.config.success_hold_seconds)
        self._timeout_seconds = float(self.config.timeout_seconds)
        self._low_z_threshold = float(self.pipette_eval.low_z_threshold)

    @property
    def last_pipette_pos(self):
        return self._last_pipette_pos

    def reset_episode(self):
        self._elapsed_s = 0.0
        self._success = False
        self._failed_low_z = False
        self._pose_backend = ""
        self._last_pipette_pos = None
        self._last_y = 0.0
        self._last_z = 0.0
        self._min_z = math.inf
        self._success_hold_s = 0.0

    def update(self, dt_seconds: float):
        if self._success or self._failed_low_z:
            return

        dt = float(dt_seconds)
        self._elapsed_s += dt

        pipette_pos, _ = self.pipette_pose_accessor.get_pose(dt_seconds=dt)
        self._pose_backend = self.pipette_pose_accessor.last_backend
        self._last_pipette_pos = pipette_pos.detach().clone() if pipette_pos is not None else None

        if pipette_pos is None:
            self._success_hold_s = 0.0
            return

        current_y = float(pipette_pos[0, 1].item())
        current_z = float(pipette_pos[0, 2].item())
        self._last_y = current_y
        self._last_z = current_z
        if current_z < self._min_z:
            self._min_z = current_z

        if current_z < self._low_z_threshold:
            self._failed_low_z = True
            self._success_hold_s = 0.0
            return

        in_success_region = (
            current_y > self._success_y_threshold
            and current_z > self._success_z_threshold
        )
        if in_success_region:
            self._success_hold_s += dt
        else:
            self._success_hold_s = 0.0

        if self._success_hold_s >= self._required_hold_s:
            self._success = True

    def _metrics(self) -> dict[str, float | str | bool]:
        return {
            "elapsed_s": self._elapsed_s,
            "pose_backend": self._pose_backend or "unknown",
            "pipette_y": self._last_y,
            "pipette_z": self._last_z,
            "min_z": self._min_z,
            "low_z_threshold": self._low_z_threshold,
            "success_y_threshold": self._success_y_threshold,
            "success_z_threshold": self._success_z_threshold,
            "success_hold_s": self._success_hold_s,
            "required_hold_s": self._required_hold_s,
            "timeout_seconds": self._timeout_seconds,
            "y_above_threshold": bool(self._last_y > self._success_y_threshold),
            "z_above_threshold": bool(self._last_z > self._success_z_threshold),
        }

    def finalize(self) -> EvalResult:
        metrics = self._metrics()
        if self._failed_low_z:
            return EvalResult(success=False, reason="pipette_z_below_threshold", metrics=metrics)

        if self._success:
            return EvalResult(success=True, reason="success", metrics=metrics)

        return EvalResult(success=False, reason="timeout", metrics=metrics)


class TakeOutPetriDishTaskEvaluator(BaseTaskEvaluator):
    name = "take_out_petri_dish"

    def __init__(self, task_preset, device: str):
        self.task_preset = task_preset
        self.device = device
        self.petri_close_prim_path = task_preset.petri_close_prim_path
        self.x_threshold = float(task_preset.petri_success_x_threshold)
        self.z_threshold = float(task_preset.petri_success_z_threshold)
        self.required_hold_s = float(task_preset.petri_success_hold_seconds)
        self.pose_accessor = PetriDishPoseAccessor(prim_path=self.petri_close_prim_path, device=device)

        self._elapsed_s = 0.0
        self._above_hold_s = 0.0
        self._success = False
        self._pose_valid = True
        self._failed_low_z = False
        self._pose_backend = ""
        self._last_x = 0.0
        self._last_z = 0.0
        self._min_z = math.inf
        self._max_z = -math.inf
        self._last_petri_close_pos = None

    @property
    def last_petri_close_pos(self):
        return self._last_petri_close_pos

    def reset_episode(self):
        self._elapsed_s = 0.0
        self._above_hold_s = 0.0
        self._success = False
        self._pose_valid = True
        self._failed_low_z = False
        self._pose_backend = ""
        self._last_x = 0.0
        self._last_z = 0.0
        self._min_z = math.inf
        self._max_z = -math.inf
        self._last_petri_close_pos = None

        petri_pos0, _ = self.pose_accessor.get_pose(dt_seconds=0.0)
        self._pose_backend = self.pose_accessor.last_backend
        if petri_pos0 is None:
            self._pose_valid = False
            return

        self._last_petri_close_pos = petri_pos0.detach().clone()
        self._last_x = float(petri_pos0[0, 0].item())
        self._last_z = float(petri_pos0[0, 2].item())
        self._min_z = self._last_z
        self._max_z = self._last_z

    def update(self, dt_seconds: float):
        if self._success or self._failed_low_z or not self._pose_valid:
            return

        self._elapsed_s += float(dt_seconds)

        petri_pos, _ = self.pose_accessor.get_pose(dt_seconds=float(dt_seconds))
        self._pose_backend = self.pose_accessor.last_backend
        if petri_pos is None:
            self._pose_valid = False
            return

        self._last_petri_close_pos = petri_pos.detach().clone()
        current_x = float(petri_pos[0, 0].item())
        current_z = float(petri_pos[0, 2].item())
        self._last_x = current_x
        self._last_z = current_z
        if current_z < self._min_z:
            self._min_z = current_z
        if current_z > self._max_z:
            self._max_z = current_z

        if current_z < self.z_threshold:
            self._failed_low_z = True
            return

        if current_x < self.x_threshold and current_z > self.z_threshold:
            self._above_hold_s += float(dt_seconds)
        else:
            self._above_hold_s = 0.0

        if self._above_hold_s >= self.required_hold_s:
            self._success = True

    def finalize(self) -> EvalResult:
        metrics = {
            "petri_close_prim_path": self.petri_close_prim_path,
            "final_x": self._last_x,
            "final_z": self._last_z,
            "min_z": self._min_z,
            "max_z": self._max_z,
            "x_threshold": self.x_threshold,
            "z_threshold": self.z_threshold,
            "above_hold_s": self._above_hold_s,
            "required_hold_s": self.required_hold_s,
            "elapsed_s": self._elapsed_s,
            "pose_backend": self._pose_backend or "unknown",
        }

        if not self._pose_valid:
            return EvalResult(success=False, reason="petri_close_pose_unavailable", metrics=metrics)

        if self._failed_low_z:
            return EvalResult(success=False, reason="petri_close_z_below_threshold", metrics=metrics)

        if self._success:
            return EvalResult(success=True, reason="success", metrics=metrics)

        return EvalResult(success=False, reason="timeout", metrics=metrics)


class PlacePetriDishTaskEvaluator(BaseTaskEvaluator):
    name = "place_petri_dish"

    def __init__(self, task_preset, device: str):
        self.task_preset = task_preset
        self.device = device
        self.petri_close_prim_path = task_preset.petri_close_prim_path
        self.x_min = float(task_preset.place_petri_success_x_min)
        self.y_max = float(task_preset.place_petri_success_y_max)
        self.z_threshold = float(task_preset.place_petri_success_z_threshold)
        self.required_stable_s = float(task_preset.place_petri_z_stable_seconds)
        self.z_stable_tolerance = float(task_preset.place_petri_z_stable_tolerance)
        self.pose_accessor = PetriDishPoseAccessor(prim_path=self.petri_close_prim_path, device=device)

        self._elapsed_s = 0.0
        self._stable_hold_s = 0.0
        self._success = False
        self._pose_valid = True
        self._failed_low_z = False
        self._pose_backend = ""
        self._last_x = 0.0
        self._last_y = 0.0
        self._last_z = 0.0
        self._prev_z = None
        self._last_z_delta = math.inf
        self._min_z = math.inf
        self._max_z = -math.inf
        self._last_petri_close_pos = None

    @property
    def last_petri_close_pos(self):
        return self._last_petri_close_pos

    def reset_episode(self):
        self._elapsed_s = 0.0
        self._stable_hold_s = 0.0
        self._success = False
        self._pose_valid = True
        self._failed_low_z = False
        self._pose_backend = ""
        self._last_x = 0.0
        self._last_y = 0.0
        self._last_z = 0.0
        self._prev_z = None
        self._last_z_delta = math.inf
        self._min_z = math.inf
        self._max_z = -math.inf
        self._last_petri_close_pos = None

        petri_pos0, _ = self.pose_accessor.get_pose(dt_seconds=0.0)
        self._pose_backend = self.pose_accessor.last_backend
        if petri_pos0 is None:
            self._pose_valid = False
            return

        self._last_petri_close_pos = petri_pos0.detach().clone()
        self._last_x = float(petri_pos0[0, 0].item())
        self._last_y = float(petri_pos0[0, 1].item())
        self._last_z = float(petri_pos0[0, 2].item())
        self._prev_z = self._last_z
        self._last_z_delta = 0.0
        self._min_z = self._last_z
        self._max_z = self._last_z

    def update(self, dt_seconds: float):
        if self._success or self._failed_low_z or not self._pose_valid:
            return

        self._elapsed_s += float(dt_seconds)

        petri_pos, _ = self.pose_accessor.get_pose(dt_seconds=float(dt_seconds))
        self._pose_backend = self.pose_accessor.last_backend
        if petri_pos is None:
            self._pose_valid = False
            return

        self._last_petri_close_pos = petri_pos.detach().clone()
        current_x = float(petri_pos[0, 0].item())
        current_y = float(petri_pos[0, 1].item())
        current_z = float(petri_pos[0, 2].item())
        self._last_x = current_x
        self._last_y = current_y
        self._last_z = current_z
        if current_z < self._min_z:
            self._min_z = current_z
        if current_z > self._max_z:
            self._max_z = current_z

        if current_z < self.z_threshold:
            self._failed_low_z = True
            return

        if self._prev_z is None:
            z_delta = math.inf
        else:
            z_delta = abs(current_z - float(self._prev_z))
        self._prev_z = current_z
        self._last_z_delta = z_delta

        in_place_region = current_y < self.y_max and current_x > self.x_min and current_z > self.z_threshold
        z_is_stable = z_delta <= self.z_stable_tolerance
        if in_place_region and z_is_stable:
            self._stable_hold_s += float(dt_seconds)
        else:
            self._stable_hold_s = 0.0

        if self._stable_hold_s >= self.required_stable_s:
            self._success = True

    def finalize(self) -> EvalResult:
        metrics = {
            "petri_close_prim_path": self.petri_close_prim_path,
            "final_x": self._last_x,
            "final_y": self._last_y,
            "final_z": self._last_z,
            "min_z": self._min_z,
            "max_z": self._max_z,
            "x_min": self.x_min,
            "y_max": self.y_max,
            "z_threshold": self.z_threshold,
            "stable_hold_s": self._stable_hold_s,
            "required_stable_s": self.required_stable_s,
            "z_stable_tolerance": self.z_stable_tolerance,
            "last_z_delta": self._last_z_delta,
            "elapsed_s": self._elapsed_s,
            "pose_backend": self._pose_backend or "unknown",
        }

        if not self._pose_valid:
            return EvalResult(success=False, reason="petri_close_pose_unavailable", metrics=metrics)

        if self._failed_low_z:
            return EvalResult(success=False, reason="petri_close_z_below_threshold", metrics=metrics)

        if self._success:
            return EvalResult(success=True, reason="success", metrics=metrics)

        return EvalResult(success=False, reason="timeout", metrics=metrics)


class PlaceTubeOnBalanceTaskEvaluator(BaseTaskEvaluator):
    name = "place_tube_on_balance"

    def __init__(self, task_preset, device: str):
        self.task_preset = task_preset
        self.device = device
        self.tube_prim_path = task_preset.tube_prim_path
        self.config = task_preset.place_tube_on_balance_eval
        self.plate_prim_path = self.config.plate_prim_path
        self.radius_scale = float(self.config.plate_radius_scale)
        self.tube_z_min = float(self.config.tube_z_min)
        self.tube_z_max = float(self.config.tube_z_max)
        self.required_hold_s = float(self.config.success_hold_seconds)
        self.pose_accessor = TubePoseAccessor(prim_path=self.tube_prim_path, device=device)

        self._elapsed_s = 0.0
        self._inside_hold_s = 0.0
        self._success = False
        self._pose_valid = True
        self._pose_backend = ""
        self._last_distance_xy = math.inf
        self._last_success_radius = 0.0
        self._last_tube_z = 0.0
        self._last_tube_pos = None
        self._last_tube_xy = None
        self._last_plate_center_xy = None

    @property
    def last_tube_pos(self):
        return self._last_tube_pos

    @property
    def last_tube_xy(self):
        return self._last_tube_xy

    def reset_episode(self):
        self._elapsed_s = 0.0
        self._inside_hold_s = 0.0
        self._success = False
        self._pose_valid = True
        self._pose_backend = ""
        self._last_distance_xy = math.inf
        self._last_success_radius = 0.0
        self._last_tube_z = 0.0
        self._last_tube_pos = None
        self._last_tube_xy = None
        self._last_plate_center_xy = None

        tube_pos0, _ = self.pose_accessor.get_pose(dt_seconds=0.0)
        self._pose_backend = self.pose_accessor.last_backend
        plate_center_xy, success_radius = _capture_prim_world_xy_circle_tensor(
            self.plate_prim_path,
            self.radius_scale,
            self.device,
        )
        if tube_pos0 is None or plate_center_xy is None or success_radius is None:
            self._pose_valid = False
            return

        self._last_tube_pos = tube_pos0.detach().clone()
        self._last_tube_xy = tube_pos0[:, :2].detach().clone()
        self._last_tube_z = float(tube_pos0[0, 2].item())
        self._last_plate_center_xy = plate_center_xy.detach().clone()
        self._last_success_radius = float(success_radius)
        self._last_distance_xy = float(torch.norm(self._last_tube_xy[0] - plate_center_xy[0]).item())

    def update(self, dt_seconds: float):
        if self._success or not self._pose_valid:
            return

        self._elapsed_s += float(dt_seconds)

        tube_pos, _ = self.pose_accessor.get_pose(dt_seconds=float(dt_seconds))
        self._pose_backend = self.pose_accessor.last_backend
        plate_center_xy, success_radius = _capture_prim_world_xy_circle_tensor(
            self.plate_prim_path,
            self.radius_scale,
            self.device,
        )
        if tube_pos is None or plate_center_xy is None or success_radius is None:
            self._pose_valid = False
            return

        tube_xy = tube_pos[:, :2]
        self._last_tube_pos = tube_pos.detach().clone()
        self._last_tube_xy = tube_xy.detach().clone()
        self._last_tube_z = float(tube_pos[0, 2].item())
        self._last_plate_center_xy = plate_center_xy.detach().clone()
        self._last_success_radius = float(success_radius)
        self._last_distance_xy = float(torch.norm(tube_xy[0] - plate_center_xy[0]).item())

        in_xy_region = self._last_distance_xy <= self._last_success_radius
        in_z_region = self.tube_z_min < self._last_tube_z < self.tube_z_max
        if in_xy_region and in_z_region:
            self._inside_hold_s += float(dt_seconds)
        else:
            self._inside_hold_s = 0.0

        if self._inside_hold_s >= self.required_hold_s:
            self._success = True

    def finalize(self) -> EvalResult:
        metrics = {
            "tube_prim_path": self.tube_prim_path,
            "plate_prim_path": self.plate_prim_path,
            "plate_radius_scale": self.radius_scale,
            "final_distance_xy": self._last_distance_xy,
            "success_radius": self._last_success_radius,
            "tube_z_min": self.tube_z_min,
            "tube_z_max": self.tube_z_max,
            "tube_z_in_range": bool(self.tube_z_min < self._last_tube_z < self.tube_z_max),
            "inside_hold_s": self._inside_hold_s,
            "required_hold_s": self.required_hold_s,
            "elapsed_s": self._elapsed_s,
            "pose_backend": self._pose_backend or "unknown",
        }

        if self._last_tube_pos is not None:
            metrics["tube_x"] = float(self._last_tube_pos[0, 0].detach().cpu().item())
            metrics["tube_y"] = float(self._last_tube_pos[0, 1].detach().cpu().item())
            metrics["tube_z"] = float(self._last_tube_pos[0, 2].detach().cpu().item())
        if self._last_tube_xy is not None:
            metrics["tube_xy"] = self._last_tube_xy[0].detach().cpu().tolist()
        if self._last_plate_center_xy is not None:
            metrics["plate_center_xy"] = self._last_plate_center_xy[0].detach().cpu().tolist()

        if not self._pose_valid:
            return EvalResult(success=False, reason="tube_or_plate_pose_unavailable", metrics=metrics)

        if self._success:
            return EvalResult(success=True, reason="success", metrics=metrics)

        return EvalResult(success=False, reason="tube_xy_not_inside_plate_circle", metrics=metrics)


def build_task_evaluator(
    task_preset,
    device: str,
    centrifuge_lid_prim_path: str = "",
    centrifuge_min_motion_deg: float = 20.0,
):
    task_id = str(task_preset.task_id)
    lower = task_id.lower()

    if lower == "place_the_pipette_on_the_pipette_stand":
        if not task_preset.pipette_prim_path:
            return NoopEvaluator()
        return PlacePipetteOnStandTaskEvaluator(task_preset=task_preset, device=device)

    if lower == "open_the_water_bath_lid":
        return WaterBathLidTaskEvaluator(task_preset=task_preset, device=device)

    if lower == "place_the_centrifuge_tube_on_the_balance":
        if not task_preset.tube_prim_path:
            return NoopEvaluator()
        return PlaceTubeOnBalanceTaskEvaluator(task_preset=task_preset, device=device)

    if "tube" in lower:
        if not task_preset.tube_prim_path:
            return NoopEvaluator()
        return TubeTaskEvaluator(task_preset=task_preset, device=device)

    if "pipette" in lower:
        return PipetteTaskEvaluator(task_preset=task_preset, device=device)

    if lower == "take_out_the_petri_dish":
        return TakeOutPetriDishTaskEvaluator(task_preset=task_preset, device=device)

    if lower == "place_the_petri_dish":
        return PlacePetriDishTaskEvaluator(task_preset=task_preset, device=device)

    if lower == "close_the_spectrophotometer":
        return CentrifugeTaskEvaluator(task_preset=task_preset, device=device)

    if "centrifuge" in lower:
        return CentrifugeTaskEvaluator(task_preset=task_preset, device=device)

    return NoopEvaluator()
