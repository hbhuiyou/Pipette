import argparse
import os
import sys
import threading
from pathlib import Path

import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PIPETTE_ROOT = SCRIPTS_DIR.parent
IMPORT_PATH_CANDIDATES = [
    SCRIPTS_DIR / "Server",
    SCRIPTS_DIR / "Data",
]
for candidate in IMPORT_PATH_CANDIDATES:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from isaaclab_env_module import (  # noqa: E402
    apply_camera_launch_workarounds,
    create_franka_simulation_session,
)
import schema as ds  # noqa: E402
from open_lid_evaluation import build_open_lid_drive_eval_result  # noqa: E402
from temporal_resampling import build_temporal_source_positions  # noqa: E402
from task_registry import DEFAULT_TASK_ID, OPEN_CENTRIFUGE_LID_TASK_ID, get_task_preset  # noqa: E402
from task_eval_logging import format_task_eval_step_logs, should_print_eval_step  # noqa: E402
from task_success_evaluator import build_task_evaluator  # noqa: E402
from virtual_button_lid_opener import VirtualButtonLidOpener, VirtualButtonLidOpenerConfig  # noqa: E402


def _next_demo_index(dst_data_group: h5py.Group) -> int:
    return ds.next_demo_index(dst_data_group)


def _normalize_last_dim(tensor: torch.Tensor, width: int, key_name: str) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)

    if tensor.shape[-1] != width:
        if tensor.numel() == width:
            tensor = tensor.reshape(1, width)
        else:
            raise RuntimeError(
                f"Dataset field '{key_name}' has invalid shape {tuple(tensor.shape)}; expected last dim {width}."
            )
    return tensor


def _load_field_tensor(group: h5py.Group, key_name: str, width: int, device: str) -> torch.Tensor:
    tensor = torch.tensor(group[key_name][:], device=device, dtype=torch.float32)
    return _normalize_last_dim(tensor, width=width, key_name=key_name)


def _step_tensor(tensor: torch.Tensor, index: int, width: int, key_name: str) -> torch.Tensor:
    step_value = tensor[index]
    if step_value.ndim == 1:
        step_value = step_value.unsqueeze(0)
    elif step_value.ndim > 2:
        step_value = step_value.reshape(step_value.shape[0], -1)

    if step_value.shape[-1] != width:
        if step_value.numel() == width:
            step_value = step_value.reshape(1, width)
        else:
            raise RuntimeError(
                f"Step tensor '{key_name}' shape mismatch: got {tuple(step_value.shape)}, expected last dim {width}."
            )
    return step_value


def _load_timestamp_series(demo_group: h5py.Group) -> list[float] | None:
    dataset_key = ds.obs_path(ds.OBS_TIMESTAMP_SIM_SEC)
    if dataset_key not in demo_group:
        return None
    values = demo_group[dataset_key][:]
    if values.size == 0:
        return None
    values = values.reshape(values.shape[0], -1)
    return [float(v[0]) for v in values]


def _parse_positive_scales(scales_text: str, argument_name: str) -> list[float]:
    items = [item.strip() for item in scales_text.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{argument_name} cannot be empty.")

    scales: list[float] = []
    for item in items:
        value = float(item)
        if value <= 0.0:
            raise ValueError(f"All values in {argument_name} must be > 0.")
        scales.append(value)
    return scales


def _dedupe_scales(scales: list[float], eps: float = 1.0e-6) -> list[float]:
    deduped: list[float] = []
    for value in scales:
        if any(abs(value - existing) <= eps for existing in deduped):
            continue
        deduped.append(float(value))
    return deduped


def _normalize_rgb_frame(rgb_tensor: torch.Tensor) -> np.ndarray:
    frame = rgb_tensor.detach().to("cpu")
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 255).to(torch.uint8)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3:
        raise RuntimeError(f"Unexpected rgb tensor shape {tuple(frame.shape)}")
    if frame.shape[-1] >= 3:
        frame = frame[..., :3]
    else:
        raise RuntimeError(f"RGB tensor last dim must be >= 3, got {tuple(frame.shape)}")
    return frame.contiguous().numpy()


def _create_rgb_dataset(dst_obs: h5py.Group, key: str, data: np.ndarray):
    if data.ndim != 4:
        raise RuntimeError(f"RGB dataset '{key}' must be 4D (T,H,W,C), got shape={tuple(data.shape)}")

    t, h, w, c = data.shape
    chunk_t = max(1, min(1, int(t)))
    dst_obs.create_dataset(
        key,
        data=data,
        dtype=np.uint8,
        chunks=(chunk_t, int(h), int(w), int(c)),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
    )


def _write_vision_freshness_datasets(dst_obs: h5py.Group, num_steps: int, vision_decimation: int):
    step_count = max(0, int(num_steps))
    decimation = max(1, int(vision_decimation))
    step_ids = np.arange(step_count, dtype=np.int32)
    fresh_flat = (step_ids % decimation) == 0

    vision_is_fresh = fresh_flat.reshape(step_count, 1).astype(np.bool_)
    last_fresh = (step_ids // decimation) * decimation
    vision_age_steps = (step_ids - last_fresh).reshape(step_count, 1).astype(np.int32)
    vision_frame_counter = (step_ids // decimation + 1).reshape(step_count, 1).astype(np.int32)

    dst_obs.create_dataset(ds.OBS_VISION_IS_FRESH, data=vision_is_fresh, dtype=np.bool_)
    dst_obs.create_dataset(ds.OBS_VISION_AGE_STEPS, data=vision_age_steps, dtype=np.int32)
    dst_obs.create_dataset(ds.OBS_VISION_FRAME_COUNTER, data=vision_frame_counter, dtype=np.int32)


def _write_recomputed_state_datasets(dst_demo: h5py.Group, state_payload: dict[str, np.ndarray]):
    dst_obs = dst_demo[ds.OBS] if ds.OBS in dst_demo else dst_demo.create_group(ds.OBS)
    dst_states = dst_demo[ds.STATES] if ds.STATES in dst_demo else dst_demo.create_group(ds.STATES)

    dst_obs.create_dataset(ds.OBS_ROBOT_JOINT_POS, data=state_payload["obs_robot_joint_pos"], dtype=np.float32)
    dst_obs.create_dataset(ds.OBS_ROBOT_JOINT_VEL, data=state_payload["obs_robot_joint_vel"], dtype=np.float32)
    dst_obs.create_dataset(ds.OBS_ROBOT_EEF_POS, data=state_payload["obs_robot_eef_pos"], dtype=np.float32)
    dst_obs.create_dataset(ds.OBS_ROBOT_EEF_QUAT, data=state_payload["obs_robot_eef_quat"], dtype=np.float32)
    dst_obs.create_dataset(ds.OBS_TIMESTAMP_SIM_SEC, data=state_payload["obs_timestamp_sim_sec"], dtype=np.float32)
    dst_obs.create_dataset(ds.OBS_TIMESTAMP_WALL_SEC, data=state_payload["obs_timestamp_wall_sec"], dtype=np.float32)

    dst_states.create_dataset(ds.STATE_ROBOT_ROOT_STATE, data=state_payload["states_robot_root_state"], dtype=np.float32)
    dst_states.create_dataset(ds.STATE_ROBOT_JOINT_POS, data=state_payload["states_robot_joint_pos"], dtype=np.float32)
    dst_states.create_dataset(ds.STATE_ROBOT_JOINT_VEL, data=state_payload["states_robot_joint_vel"], dtype=np.float32)


def _copy_group_tree(src_group: h5py.Group, dst_group: h5py.Group, excluded_paths: set[str], prefix: str = ""):
    for key in src_group.keys():
        src_obj = src_group[key]
        rel_path = f"{prefix}/{key}" if prefix else key
        if rel_path in excluded_paths:
            continue

        if isinstance(src_obj, h5py.Group):
            child_dst = dst_group.create_group(key)
            for attr_key, attr_value in src_obj.attrs.items():
                child_dst.attrs[attr_key] = attr_value
            _copy_group_tree(src_obj, child_dst, excluded_paths, prefix=rel_path)
        else:
            src_group.copy(src_obj, dst_group, name=key)


def _resolve_output_file(input_file: str, output_file: str) -> str:
    if output_file:
        return output_file
    root, ext = os.path.splitext(input_file)
    if ext.lower() not in {".h5", ".hdf5"}:
        ext = ".hdf5"
    return f"{root}_aug_light{ext}"


def _collect_light_intensity_bases() -> list[tuple[object, float]]:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    bases: list[tuple[object, float]] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        intensity_attr = prim.GetAttribute("inputs:intensity")
        if not intensity_attr.IsValid():
            intensity_attr = prim.GetAttribute("intensity")
            if not intensity_attr.IsValid():
                continue

        base_intensity = intensity_attr.Get()
        if base_intensity is None:
            continue

        try:
            bases.append((intensity_attr, float(base_intensity)))
        except Exception:
            continue

    return bases


def _set_light_intensity_scale(light_bases: list[tuple[object, float]], scale: float) -> int:
    light_count = 0
    for intensity_attr, base_intensity in light_bases:
        intensity_attr.Set(float(base_intensity) * float(scale))
        light_count += 1
    return light_count


def _scale_demo_timestamps_inplace(demo_group: h5py.Group, speed_scale: float):
    if abs(float(speed_scale) - 1.0) <= 1.0e-6:
        return

    for dataset_key in (
        ds.obs_path(ds.OBS_TIMESTAMP_SIM_SEC),
        ds.obs_path(ds.OBS_TIMESTAMP_WALL_SEC),
    ):
        if dataset_key not in demo_group:
            continue

        ds = demo_group[dataset_key]
        values = ds[:]
        if values.size == 0:
            continue

        values_2d = values.reshape(values.shape[0], -1).astype(np.float64, copy=False)
        base = values_2d[0:1, :]
        scaled = base + (values_2d - base) / float(speed_scale)
        ds[...] = scaled.reshape(values.shape).astype(ds.dtype, copy=False)


def _interp_1d_series(values: np.ndarray, x: float) -> float:
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])

    x_clamped = float(np.clip(x, 0.0, float(values.size - 1)))
    left = int(np.floor(x_clamped))
    right = min(left + 1, values.size - 1)
    alpha = float(x_clamped - left)
    return float((1.0 - alpha) * float(values[left]) + alpha * float(values[right]))


def _interp_action_tensor(actions: torch.Tensor, x: float) -> torch.Tensor:
    num_steps = int(actions.shape[0])
    if num_steps <= 1:
        return actions[0:1].clone()

    x_clamped = float(np.clip(x, 0.0, float(num_steps - 1)))
    left = int(np.floor(x_clamped))
    right = min(left + 1, num_steps - 1)
    alpha = float(x_clamped - left)
    action = (1.0 - alpha) * actions[left:left + 1] + alpha * actions[right:right + 1]
    return action


def _interp_action_tensor_cubic(actions: torch.Tensor, x: float) -> torch.Tensor:
    num_steps = int(actions.shape[0])
    if num_steps <= 1:
        return actions[0:1].clone()

    x_clamped = float(np.clip(x, 0.0, float(num_steps - 1)))
    i1 = int(np.floor(x_clamped))
    i2 = min(i1 + 1, num_steps - 1)
    i0 = max(i1 - 1, 0)
    i3 = min(i2 + 1, num_steps - 1)
    t = float(x_clamped - i1)

    p0 = actions[i0:i0 + 1]
    p1 = actions[i1:i1 + 1]
    p2 = actions[i2:i2 + 1]
    p3 = actions[i3:i3 + 1]

    # Catmull-Rom spline in tensor form.
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _sample_offset_in_sphere(rng: np.random.Generator, radius_m: float) -> tuple[float, float, float]:
    # Uniform sample in 3D ball: direction from normal, radius from U^(1/3).
    direction = rng.normal(0.0, 1.0, size=3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        return (0.0, 0.0, 0.0)
    direction = direction / norm
    scale = float(radius_m) * float(rng.random() ** (1.0 / 3.0))
    offset = direction * scale
    return (float(offset[0]), float(offset[1]), float(offset[2]))


def _build_camera_jitter_specs(
    camera_names: list[str],
    jitter_count: int,
    radius_m: float,
    camera_radius_overrides_m: dict[str, float] | None = None,
) -> list[tuple[str, dict[str, tuple[float, float, float]] | None]]:
    if int(jitter_count) <= 0:
        return [("none", None)]

    rng = np.random.default_rng()
    radius_overrides = dict(camera_radius_overrides_m or {})
    specs: list[tuple[str, dict[str, tuple[float, float, float]] | None]] = []
    for jitter_idx in range(int(jitter_count)):
        offsets: dict[str, tuple[float, float, float]] = {}
        for cam_name in camera_names:
            cam_radius_m = float(radius_overrides.get(cam_name, radius_m))
            offsets[cam_name] = _sample_offset_in_sphere(rng=rng, radius_m=cam_radius_m)
        specs.append((f"jitter_{jitter_idx}", offsets))
    return specs


def _close_simulation_app_with_timeout(sim_app, timeout_sec: float, force_exit_code: int):
    if sim_app is None:
        return

    timeout = float(timeout_sec)
    if timeout <= 0.0:
        sim_app.close()
        return

    closed_event = threading.Event()

    def _close_watchdog():
        if closed_event.wait(timeout):
            return
        print(
            f"[WARN] simulation_app.close() timed out after {timeout:.1f}s. "
            f"Force exiting process with code {int(force_exit_code)}.",
            flush=True,
        )
        os._exit(int(force_exit_code))

    watchdog = threading.Thread(target=_close_watchdog, name="sim-close-watchdog", daemon=True)
    watchdog.start()
    sim_app.close()
    closed_event.set()


def main():
    parser = argparse.ArgumentParser(description="Offline dataset augmentation by replaying trajectories under different light intensity.")
    parser.add_argument("--dataset_file", type=str, default="", help="Input HDF5 dataset path. Empty means task preset default.")
    parser.add_argument("--output_file", type=str, default="", help="Output HDF5 path. Default: <dataset>_aug_light.hdf5")
    parser.add_argument("--task_id", type=str, default=DEFAULT_TASK_ID)
    parser.add_argument("--camera_width", type=int, default=400)
    parser.add_argument("--camera_height", type=int, default=400)
    parser.add_argument(
        "--demo_indices",
        type=str,
        default="",
        help="Optional comma-separated demo indices (e.g. '0,2,5'). Empty means all demos.",
    )
    parser.add_argument(
        "--light_intensity_scales",
        type=str,
        default="0.8",
        help="Comma-separated scales, e.g. '0.6' or '0.6,1.4'.",
    )
    parser.add_argument(
        "--temporal_speed_scales",
        type=str,
        default="1.2",
        help="Comma-separated temporal speed scales, e.g. '1.0,0.8,1.2'. >1 faster, <1 slower.",
    )
    parser.add_argument(
        "--temporal_interp_mode",
        type=str,
        choices=["linear", "cubic"],
        default="cubic",
        help="Action interpolation mode along temporal resampling path.",
    )
    parser.add_argument(
        "--temporal_warp_strength",
        type=float,
        default=0.10,
        help="Non-uniform temporal warp strength in [0,1]. 0 disables random local speed changes.",
    )
    parser.add_argument(
        "--temporal_warp_smooth_window",
        type=int,
        default=11,
        help="Smoothing window for temporal warp profile.",
    )
    parser.add_argument(
        "--action_jitter_std",
        type=float,
        default=0.005,
        help="Gaussian std of local arm-joint jitter (radians) added to interpolated actions.",
    )
    parser.add_argument(
        "--action_jitter_clip",
        type=float,
        default=0.02,
        help="Absolute clamp for arm-joint jitter delta (radians).",
    )
    parser.add_argument(
        "--include_original",
        action="store_true",
        help="Also copy original demos into output dataset (no light/temporal augmentation).",
    )
    parser.add_argument(
        "--camera_jitter_count",
        type=int,
        default=5,
        help=(
            "Number of non-zero candidate camera jitter poses to sample. "
            "If >0, each base variant writes "
            "one non-jittered replay plus one replay using a randomly selected candidate."
        ),
    )
    parser.add_argument(
        "--close_timeout_sec",
        type=float,
        default=5.0,
        help="Timeout in seconds for simulation_app.close(). <=0 disables force-exit watchdog.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    if int(args_cli.camera_jitter_count) < 0:
        raise ValueError("--camera_jitter_count must be >= 0.")
    if float(args_cli.temporal_warp_strength) < 0.0:
        raise ValueError("--temporal_warp_strength must be >= 0.")
    if float(args_cli.action_jitter_std) < 0.0:
        raise ValueError("--action_jitter_std must be >= 0.")
    if float(args_cli.action_jitter_clip) < 0.0:
        raise ValueError("--action_jitter_clip must be >= 0.")

    task_preset = get_task_preset(args_cli.task_id)
    if not args_cli.dataset_file:
        args_cli.dataset_file = task_preset.dataset_file
    args_cli.output_file = _resolve_output_file(args_cli.dataset_file, args_cli.output_file)
    light_scales = _dedupe_scales(_parse_positive_scales(args_cli.light_intensity_scales, "--light_intensity_scales"))
    temporal_speed_scales = _dedupe_scales(
        _parse_positive_scales(args_cli.temporal_speed_scales, "--temporal_speed_scales")
    )

    variant_specs: list[tuple[str, float, float]] = []
    if args_cli.include_original:
        variant_specs.append(("original", 1.0, 1.0))
    for light_scale in light_scales:
        if abs(float(light_scale) - 1.0) <= 1.0e-6:
            continue
        variant_specs.append(("light_only", float(light_scale), 1.0))
    for speed_scale in temporal_speed_scales:
        if abs(float(speed_scale) - 1.0) <= 1.0e-6:
            continue
        variant_specs.append(("temporal_only", 1.0, float(speed_scale)))
    for light_scale in light_scales:
        if abs(float(light_scale) - 1.0) <= 1.0e-6:
            continue
        for speed_scale in temporal_speed_scales:
            if abs(float(speed_scale) - 1.0) <= 1.0e-6:
                continue
            variant_specs.append(("light_and_temporal", float(light_scale), float(speed_scale)))
    if not variant_specs and int(args_cli.camera_jitter_count) > 0:
        variant_specs.append(("camera_only", 1.0, 1.0))

    if not variant_specs:
        raise ValueError(
            "No output variants selected. Enable --include_original and/or provide valid "
            "--light_intensity_scales/--temporal_speed_scales/--camera_jitter_count."
        )

    args_cli = apply_camera_launch_workarounds(args_cli)
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    camera_width = max(32, int(args_cli.camera_width))
    camera_height = max(32, int(args_cli.camera_height))
    camera_paths = {spec.name: spec.prim_path for spec in task_preset.camera_specs}
    for required_camera in ("main", "top", "wrist"):
        if required_camera not in camera_paths:
            raise ValueError(f"Task preset '{task_preset.task_id}' misses required camera '{required_camera}'.")

    control_hz = max(1, int(task_preset.control_hz))
    vision_hz = max(1, min(int(task_preset.vision_hz), control_hz))
    vision_decimation = max(1, int(round(control_hz / vision_hz)))
    camera_jitter_radius_m = 0.01
    camera_jitter_radii_m = {
        spec.name: (0.002 if spec.name == "wrist" else 0.2)
        for spec in task_preset.camera_specs
    }
    camera_jitter_specs = _build_camera_jitter_specs(
        camera_names=[spec.name for spec in task_preset.camera_specs],
        jitter_count=int(args_cli.camera_jitter_count),
        radius_m=float(camera_jitter_radius_m),
        camera_radius_overrides_m=camera_jitter_radii_m,
    )
    camera_jitter_enabled = camera_jitter_specs[0][1] is not None
    variants_per_demo = sum(
        1 if variant_type == "camera_only" and camera_jitter_enabled else (2 if camera_jitter_enabled else 1)
        for variant_type, _, _ in variant_specs
    )

    print(f"[INFO] Input dataset: {args_cli.dataset_file}")
    print(f"[INFO] Output dataset: {args_cli.output_file}")
    print(f"[INFO] Light intensity scales: {light_scales}")
    print(f"[INFO] Temporal speed scales: {temporal_speed_scales}")
    print(
        f"[INFO] Temporal enhancement: interp={args_cli.temporal_interp_mode}, "
        f"warp_strength={float(args_cli.temporal_warp_strength):.4f}, "
        f"warp_window={int(args_cli.temporal_warp_smooth_window)}, "
        f"action_jitter_std={float(args_cli.action_jitter_std):.4f}, "
        f"action_jitter_clip={float(args_cli.action_jitter_clip):.4f}"
    )
    print(
        f"[INFO] Camera jitter: count={int(args_cli.camera_jitter_count)}, "
        f"base_radius_m={camera_jitter_radius_m:.4f}, "
        f"per_camera_radius_m={camera_jitter_radii_m}"
    )
    print("[INFO] RGB HDF5 compression: gzip(level=4), chunks=(1,H,W,C), shuffle=True")
    print(
        f"[INFO] Fresh vision cadence: control_hz={control_hz}, "
        f"vision_hz={vision_hz}, vision_decimation={vision_decimation}"
    )
    print(
        f"[INFO] Variant selection: include_original={args_cli.include_original}, "
        f"total_variants_per_demo={variants_per_demo}"
    )

    if not os.path.exists(args_cli.dataset_file):
        print(f"[ERROR] Dataset file not found: {args_cli.dataset_file}")
        simulation_app.close()
        raise SystemExit(1)

    if os.path.abspath(args_cli.dataset_file) == os.path.abspath(args_cli.output_file):
        print("[ERROR] --output_file must be different from --dataset_file.")
        simulation_app.close()
        raise SystemExit(1)

    output_dir = os.path.dirname(os.path.abspath(args_cli.output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sim_dt = 1.0 / 60.0
    session = create_franka_simulation_session(
        task_preset,
        camera_width=camera_width,
        camera_height=camera_height,
        camera_sensor_type="camera",
        warmup_render_steps=6,
        sim_dt=sim_dt,
        render_interval=1,
        enable_sensor_capture=True,
        find_ee_body=True,
    )
    env_module = session.env_module
    sim = session.sim
    robot = session.robot
    device = sim.device

    arm_joint_ids = session.arm_joint_ids
    gripper_joint_ids = session.gripper_joint_ids
    ee_body_id = session.ee_body_id
    gripper_open_target = torch.full((1, 2), 0.04, dtype=torch.float32, device=device)
    gripper_close_target = torch.zeros((1, 2), dtype=torch.float32, device=device)
    gripper_button_body_ids = []
    for gripper_body_name in ("panda_leftfinger", "panda_rightfinger"):
        try:
            body_ids = robot.find_bodies(gripper_body_name)[0]
        except Exception:
            body_ids = []
        if body_ids:
            gripper_button_body_ids.append(body_ids[0])


    def get_virtual_button_probe_positions():
        if gripper_button_body_ids:
            finger_pos = robot.data.body_pos_w[:, gripper_button_body_ids]
            probe_pos = [finger_pos[:, index, :] for index in range(finger_pos.shape[1])]
            probe_pos.append(finger_pos.mean(dim=1))
            probe_pos.append(robot.data.body_pos_w[:, ee_body_id])
            return torch.cat(probe_pos, dim=0)
        return robot.data.body_pos_w[:, ee_body_id]


    virtual_button_lid_opener = None
    if task_preset.task_id == OPEN_CENTRIFUGE_LID_TASK_ID:
        virtual_button_lid_opener = VirtualButtonLidOpener(
            VirtualButtonLidOpenerConfig(
                button_prim_path=task_preset.centrifuge_eval.virtual_button_prim_path,
                joint_prim_path=task_preset.centrifuge_eval.lid_joint_prim_path,
            ),
            device=device,
        )
        virtual_button_lid_opener.initialize()

    sensor_cameras = session.sensor_cameras
    for required_camera in ("main", "top", "wrist"):
        if required_camera not in sensor_cameras:
            raise RuntimeError(f"Sensor camera '{required_camera}' is unavailable.")

    light_bases = _collect_light_intensity_bases()
    if not light_bases:
        print("[WARN] No UsdLux light intensity attributes found in current stage.")

    fallback_control_decimation = max(1, int(round((1.0 / control_hz) / sim_dt)))


    def _collect_augmented_rgb_frames(
        num_steps: int,
        actions: torch.Tensor,
        timestamp_series: list[float] | None,
        speed_scale: float,
        interp_mode: str,
        temporal_warp_strength: float,
        temporal_warp_smooth_window: int,
        action_jitter_std: float,
        action_jitter_clip: float,
        init_root: torch.Tensor,
        init_jpos: torch.Tensor,
        init_jvel: torch.Tensor,
        evaluator,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], object]:
        rgb_top_frames: list[np.ndarray] = []
        rgb_main_frames: list[np.ndarray] = []
        rgb_wrist_frames: list[np.ndarray] = []
        executed_actions: list[np.ndarray] = []

        obs_robot_joint_pos: list[np.ndarray] = []
        obs_robot_joint_vel: list[np.ndarray] = []
        obs_robot_eef_pos: list[np.ndarray] = []
        obs_robot_eef_quat: list[np.ndarray] = []
        obs_timestamp_sim_sec: list[np.ndarray] = []
        obs_timestamp_wall_sec: list[np.ndarray] = []

        states_robot_root_state: list[np.ndarray] = []
        states_robot_joint_pos: list[np.ndarray] = []
        states_robot_joint_vel: list[np.ndarray] = []

        sim_elapsed_sec = 0.0

        speed = float(max(1.0e-6, speed_scale))
        source_positions = build_temporal_source_positions(
            num_source_steps=num_steps,
            speed_scale=speed,
            warp_strength=float(temporal_warp_strength),
            warp_smooth_window=int(temporal_warp_smooth_window),
            rng=rng,
        )
        output_steps = int(source_positions.size)

        timestamp_array = None
        if timestamp_series is not None:
            timestamp_array = np.asarray(timestamp_series, dtype=np.float64).reshape(-1)

        if virtual_button_lid_opener is not None:
            virtual_button_lid_opener.reset_to_default_closed_state()

        sim.reset()
        robot.reset()
        for sensor in sensor_cameras.values():
            sensor.reset()

        robot.write_root_pose_to_sim(init_root[:, :7])
        robot.write_root_velocity_to_sim(init_root[:, 7:])
        robot.write_joint_state_to_sim(init_jpos, init_jvel)
        sim.step(render=True)
        robot.update(sim_dt)

        open_lid_drive_eval = task_preset.task_id == OPEN_CENTRIFUGE_LID_TASK_ID
        drive_target_initial = None
        drive_target_min = None
        drive_target_max = None
        drive_target_current = None
        if open_lid_drive_eval and virtual_button_lid_opener is not None:
            drive_target_initial = virtual_button_lid_opener.get_drive_target_position()
            drive_target_current = drive_target_initial
            drive_target_min = drive_target_initial
            drive_target_max = drive_target_initial

        if evaluator is not None:
            evaluator.reset_episode()

        for i, source_position in enumerate(source_positions):
            if not simulation_app.is_running():
                raise RuntimeError("Simulation app closed during augmentation.")

            src_index = float(source_position)
            if interp_mode == "cubic":
                action = _interp_action_tensor_cubic(actions, src_index)
            else:
                action = _interp_action_tensor(actions, src_index)
            action = _normalize_last_dim(action, width=8, key_name=ds.ACTIONS)
            arm_targets = action[:, :7].clone()

            jitter_std = float(max(0.0, action_jitter_std))
            if jitter_std > 0.0:
                noise = rng.normal(0.0, jitter_std, size=(1, 7)).astype(np.float32)
                jitter = torch.tensor(noise, dtype=torch.float32, device=arm_targets.device)
                clip_abs = float(max(0.0, action_jitter_clip))
                if clip_abs > 0.0:
                    jitter = jitter.clamp(min=-clip_abs, max=clip_abs)
                arm_targets = arm_targets + jitter

            gripper_cmd = bool(action[:, 7].item() > 0.5)
            gripper_numeric = torch.tensor([[1.0 if gripper_cmd else 0.0]], dtype=torch.float32, device=arm_targets.device)
            executed_action = torch.cat([arm_targets, gripper_numeric], dim=-1)

            # Record observation before executing the paired action, preserving
            # behavior-cloning causality: obs[t] -> action[t].
            gripper_state = robot.data.joint_pos[:, gripper_joint_ids[0:1]].clone()
            obs_joint_pos = torch.cat([robot.data.joint_pos[:, arm_joint_ids].clone(), gripper_state], dim=-1)
            obs_joint_vel = robot.data.joint_vel[:, arm_joint_ids].clone()
            obs_eef_pos = robot.data.body_pos_w[:, ee_body_id].clone()
            obs_eef_quat = robot.data.body_quat_w[:, ee_body_id].clone()
            ts_sim = torch.tensor([[sim_elapsed_sec]], dtype=torch.float32, device=device)
            ts_wall = torch.tensor([[sim_elapsed_sec]], dtype=torch.float32, device=device)

            obs_robot_joint_pos.append(obs_joint_pos.detach().to("cpu").numpy().astype(np.float32, copy=True))
            obs_robot_joint_vel.append(obs_joint_vel.detach().to("cpu").numpy().astype(np.float32, copy=True))
            obs_robot_eef_pos.append(obs_eef_pos.detach().to("cpu").numpy().astype(np.float32, copy=True))
            obs_robot_eef_quat.append(obs_eef_quat.detach().to("cpu").numpy().astype(np.float32, copy=True))
            obs_timestamp_sim_sec.append(ts_sim.detach().to("cpu").numpy().astype(np.float32, copy=True))
            obs_timestamp_wall_sec.append(ts_wall.detach().to("cpu").numpy().astype(np.float32, copy=True))

            states_robot_root_state.append(
                robot.data.root_state_w[:, :13].detach().to("cpu").numpy().astype(np.float32, copy=True)
            )
            states_robot_joint_pos.append(robot.data.joint_pos.detach().to("cpu").numpy().astype(np.float32, copy=True))
            states_robot_joint_vel.append(robot.data.joint_vel.detach().to("cpu").numpy().astype(np.float32, copy=True))

            rgb_top = env_module.capture_rgb("top", sim_dt)
            rgb_main = env_module.capture_rgb("main", sim_dt)
            rgb_wrist = env_module.capture_rgb("wrist", sim_dt)
            if rgb_top is None or rgb_main is None or rgb_wrist is None:
                raise RuntimeError(f"RGB capture failed at step {i}.")

            rgb_top_frames.append(_normalize_rgb_frame(rgb_top))
            rgb_main_frames.append(_normalize_rgb_frame(rgb_main))
            rgb_wrist_frames.append(_normalize_rgb_frame(rgb_wrist))
            executed_actions.append(executed_action.detach().to("cpu").numpy()[0].astype(np.float32, copy=True))

            gripper_targets = gripper_open_target if gripper_cmd else gripper_close_target
            robot.set_joint_position_target(arm_targets, joint_ids=arm_joint_ids)
            robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
            robot.write_data_to_sim()

            if timestamp_array is not None and i > 0:
                t_prev = _interp_1d_series(timestamp_array, float(source_positions[i - 1]))
                t_curr = _interp_1d_series(timestamp_array, src_index)
                dt_control = max(0.0, (t_curr - t_prev) / speed)
                sim_steps = max(1, int(round(dt_control / sim_dt)))
            else:
                sim_steps = fallback_control_decimation

            for _ in range(sim_steps):
                sim.step(render=True)
                robot.update(sim_dt)
                if virtual_button_lid_opener is not None:
                    virtual_button_lid_opener.update(get_virtual_button_probe_positions(), sim_dt)
                    if open_lid_drive_eval:
                        drive_target = virtual_button_lid_opener.get_drive_target_position()
                        if drive_target is not None:
                            drive_target_current = float(drive_target)
                            drive_target_min = (
                                float(drive_target)
                                if drive_target_min is None
                                else min(float(drive_target_min), float(drive_target))
                            )
                            drive_target_max = (
                                float(drive_target)
                                if drive_target_max is None
                                else max(float(drive_target_max), float(drive_target))
                            )

            sim_elapsed_sec += float(sim_steps) * float(sim_dt)

            if open_lid_drive_eval:
                step_eval_result = build_open_lid_drive_eval_result(
                    initial_target_deg=drive_target_initial,
                    min_target_deg=drive_target_min,
                    max_target_deg=drive_target_max,
                    triggered=bool(
                        virtual_button_lid_opener is not None
                        and virtual_button_lid_opener.has_triggered
                    ),
                    success_threshold_deg=float(task_preset.centrifuge_eval.drive_target_success_threshold_deg),
                )
            else:
                evaluator.update(sim_dt * sim_steps)
                step_eval_result = evaluator.finalize()

            if should_print_eval_step(i, interval_steps=30, total_steps=output_steps):
                for log_line in format_task_eval_step_logs(
                    task_preset=task_preset,
                    eval_result=step_eval_result,
                    evaluator=evaluator,
                    step_index=i,
                    total_steps=output_steps,
                    drive_target_current_deg=drive_target_current,
                ):
                    print(log_line)

        if open_lid_drive_eval:
            eval_result = build_open_lid_drive_eval_result(
                initial_target_deg=drive_target_initial,
                min_target_deg=drive_target_min,
                max_target_deg=drive_target_max,
                triggered=bool(
                    virtual_button_lid_opener is not None
                    and virtual_button_lid_opener.has_triggered
                ),
                success_threshold_deg=float(task_preset.centrifuge_eval.drive_target_success_threshold_deg),
            )
        else:
            eval_result = evaluator.finalize()

        return (
            np.stack(rgb_top_frames),
            np.stack(rgb_main_frames),
            np.stack(rgb_wrist_frames),
            np.stack(executed_actions),
            {
                "obs_robot_joint_pos": np.stack(obs_robot_joint_pos),
                "obs_robot_joint_vel": np.stack(obs_robot_joint_vel),
                "obs_robot_eef_pos": np.stack(obs_robot_eef_pos),
                "obs_robot_eef_quat": np.stack(obs_robot_eef_quat),
                "obs_timestamp_sim_sec": np.stack(obs_timestamp_sim_sec),
                "obs_timestamp_wall_sec": np.stack(obs_timestamp_wall_sec),
                "states_robot_root_state": np.stack(states_robot_root_state),
                "states_robot_joint_pos": np.stack(states_robot_joint_pos),
                "states_robot_joint_vel": np.stack(states_robot_joint_vel),
            },
            eval_result,
        )


    try:
        dst_exists = os.path.exists(args_cli.output_file)
        dst_mode = "a" if dst_exists else "w"

        with h5py.File(args_cli.dataset_file, "r") as src_h5, h5py.File(args_cli.output_file, dst_mode) as dst_h5:
            if not dst_exists:
                for attr_key, attr_value in src_h5.attrs.items():
                    dst_h5.attrs[attr_key] = attr_value
            dst_h5.attrs["augmentation_type"] = "light_intensity_and_temporal_speed"
            dst_h5.attrs["augmentation_light_scales"] = np.array(light_scales, dtype=np.float32)
            dst_h5.attrs["augmentation_temporal_speed_scales"] = np.array(temporal_speed_scales, dtype=np.float32)
            dst_h5.attrs["augmentation_temporal_interp_mode"] = str(args_cli.temporal_interp_mode)
            dst_h5.attrs["augmentation_temporal_warp_strength"] = float(args_cli.temporal_warp_strength)
            dst_h5.attrs["augmentation_temporal_warp_smooth_window"] = int(args_cli.temporal_warp_smooth_window)
            dst_h5.attrs["augmentation_action_jitter_std"] = float(args_cli.action_jitter_std)
            dst_h5.attrs["augmentation_action_jitter_clip"] = float(args_cli.action_jitter_clip)
            dst_h5.attrs["augmentation_include_original"] = bool(args_cli.include_original)
            dst_h5.attrs["augmentation_source_file"] = os.path.abspath(args_cli.dataset_file)
            dst_h5.attrs["augmentation_action_only"] = True
            dst_h5.attrs["augmentation_discard_failed"] = True
            dst_h5.attrs["augmentation_camera_jitter_count"] = int(args_cli.camera_jitter_count)
            dst_h5.attrs["augmentation_camera_jitter_candidate_count"] = int(len(camera_jitter_specs))
            dst_h5.attrs["augmentation_camera_jitter_policy"] = (
                "nonzero_candidate_pool_random_one_plus_none"
                if int(args_cli.camera_jitter_count) > 0
                else "disabled"
            )
            dst_h5.attrs["augmentation_camera_jitter_radius_m"] = float(camera_jitter_radius_m)
            for cam_name, cam_radius_m in camera_jitter_radii_m.items():
                dst_h5.attrs[f"augmentation_camera_jitter_{cam_name}_radius_m"] = float(cam_radius_m)
            dst_h5.attrs["augmentation_rgb_compression"] = "gzip"
            dst_h5.attrs["augmentation_rgb_compression_level"] = int(4)
            dst_h5.attrs["augmentation_rgb_chunk_layout"] = "(1,H,W,C)"
            dst_h5.attrs["augmentation_rgb_shuffle"] = True

            src_data = ds.require_data_group(src_h5)
            dst_data = ds.require_or_create_data_group(dst_h5)

            if dst_exists:
                print(f"[INFO] Append mode enabled: {args_cli.output_file}")
            else:
                print(f"[INFO] Create mode enabled: {args_cli.output_file}")

            available_demo_names = ds.sorted_demo_names(src_data)
            available_demo_name_set = set(available_demo_names)

            if args_cli.demo_indices.strip():
                requested_indices: list[int] = []
                for token in str(args_cli.demo_indices).split(","):
                    token = token.strip()
                    if not token:
                        continue
                    requested_indices.append(int(token))

                demos: list[str] = []
                for idx in requested_indices:
                    selected = ds.demo_name(int(idx))
                    if selected not in available_demo_name_set:
                        raise KeyError(f"{selected} not found in dataset")
                    demos.append(selected)
            else:
                demos = available_demo_names

            if not demos:
                raise RuntimeError("No demos found in input dataset.")

            print(f"[INFO] Selected {len(demos)} input demo(s).")

            out_demo_index = _next_demo_index(dst_data)
            print(f"[INFO] Output demo id starts from: demo_{out_demo_index}")
            kept_variant_count = 0
            dropped_variant_count = 0
            camera_jitter_choice_rng = np.random.default_rng()
            for src_demo_name in demos:
                src_demo = src_data[src_demo_name]
                if not isinstance(src_demo, h5py.Group):
                    continue

                actions = _load_field_tensor(src_demo, ds.ACTIONS, width=8, device=device)
                num_steps = int(actions.shape[0])
                if num_steps == 0:
                    print(f"[WARN] {src_demo_name} has zero steps, skipping.")
                    continue

                init_root = _load_field_tensor(src_demo, ds.initial_path(ds.INITIAL_ROBOT_ROOT_STATE), width=13, device=device)
                init_jpos = _load_field_tensor(src_demo, ds.initial_path(ds.INITIAL_ROBOT_JOINT_POS), width=9, device=device)
                init_jvel = _load_field_tensor(src_demo, ds.initial_path(ds.INITIAL_ROBOT_JOINT_VEL), width=9, device=device)

                timestamp_series = _load_timestamp_series(src_demo)

                print(f"[INFO] Source demo: {src_demo_name}, steps={num_steps}")
                for variant_type, light_scale, speed_scale in variant_specs:
                    active_camera_jitter_specs = [] if variant_type == "camera_only" else [("none", None)]
                    if camera_jitter_enabled:
                        selected_jitter_idx = int(camera_jitter_choice_rng.integers(0, len(camera_jitter_specs)))
                        active_camera_jitter_specs.append(camera_jitter_specs[selected_jitter_idx])

                    for jitter_name, jitter_offsets in active_camera_jitter_specs:
                        for cam_spec in task_preset.camera_specs:
                            cam_offset = None
                            if jitter_offsets is not None:
                                cam_offset = jitter_offsets.get(cam_spec.name)
                            env_module.refresh_camera_prim(cam_spec.name, translation_offset=cam_offset)

                        light_count = _set_light_intensity_scale(light_bases, light_scale)
                        if light_count == 0:
                            print(f"[WARN] No UsdLux lights found when applying light scale={light_scale:.4f}.")

                        evaluator = None
                        if task_preset.task_id != OPEN_CENTRIFUGE_LID_TASK_ID:
                            evaluator = build_task_evaluator(
                                task_preset=task_preset,
                                device=device,
                            )

                        variant_seed = abs(hash((src_demo_name, variant_type, float(light_scale), float(speed_scale), jitter_name))) % (2**32)
                        rng = np.random.default_rng(variant_seed)

                        rgb_top, rgb_main, rgb_wrist, executed_actions, state_payload, eval_result = _collect_augmented_rgb_frames(
                            num_steps=num_steps,
                            actions=actions,
                            timestamp_series=timestamp_series,
                            speed_scale=float(speed_scale),
                            interp_mode=str(args_cli.temporal_interp_mode),
                            temporal_warp_strength=float(args_cli.temporal_warp_strength),
                            temporal_warp_smooth_window=int(args_cli.temporal_warp_smooth_window),
                            action_jitter_std=float(args_cli.action_jitter_std),
                            action_jitter_clip=float(args_cli.action_jitter_clip),
                            init_root=init_root,
                            init_jpos=init_jpos,
                            init_jvel=init_jvel,
                            evaluator=evaluator,
                            rng=rng,
                        )

                        if not bool(eval_result.success):
                            dropped_variant_count += 1
                            print(
                                f"[INFO] Dropped variant from {src_demo_name}: "
                                f"variant={variant_type}, light={light_scale:.4f}, speed={speed_scale:.4f}, "
                                f"camera_jitter={jitter_name}, reason={eval_result.reason}"
                            )
                            continue

                        kept_variant_count += 1

                        out_demo_name = ds.demo_name(out_demo_index)
                        out_demo_index += 1
                        dst_demo = dst_data.create_group(out_demo_name)

                        for attr_key, attr_value in src_demo.attrs.items():
                            dst_demo.attrs[attr_key] = attr_value
                        dst_demo.attrs["augmentation_source_demo"] = src_demo_name
                        dst_demo.attrs["augmentation_variant"] = variant_type
                        dst_demo.attrs["augmentation_light_intensity_scale"] = float(light_scale)
                        dst_demo.attrs["augmentation_temporal_speed_scale"] = float(speed_scale)
                        dst_demo.attrs["augmentation_source_steps"] = int(num_steps)
                        dst_demo.attrs["augmentation_output_steps"] = int(executed_actions.shape[0])
                        dst_demo.attrs["augmentation_temporal_interp_mode"] = str(args_cli.temporal_interp_mode)
                        dst_demo.attrs["augmentation_temporal_warp_strength"] = float(args_cli.temporal_warp_strength)
                        dst_demo.attrs["augmentation_temporal_warp_smooth_window"] = int(args_cli.temporal_warp_smooth_window)
                        dst_demo.attrs["augmentation_action_jitter_std"] = float(args_cli.action_jitter_std)
                        dst_demo.attrs["augmentation_action_jitter_clip"] = float(args_cli.action_jitter_clip)
                        dst_demo.attrs["augmentation_camera_jitter"] = str(jitter_name)
                        dst_demo.attrs["augmentation_eval_success"] = True
                        dst_demo.attrs["augmentation_eval_reason"] = str(eval_result.reason)
                        if task_preset.task_id == OPEN_CENTRIFUGE_LID_TASK_ID:
                            dst_demo.attrs["augmentation_eval_type"] = "open_lid_drive_target_position"
                        elif hasattr(evaluator, "name"):
                            dst_demo.attrs["augmentation_eval_type"] = str(evaluator.name)
                        for metric_key, metric_value in dict(eval_result.metrics).items():
                            attr_key = f"augmentation_eval_{metric_key}"
                            if isinstance(metric_value, (int, float, np.integer, np.floating, str, bytes, bool)):
                                dst_demo.attrs[attr_key] = metric_value
                        if jitter_offsets is not None:
                            for cam_name, cam_offset in jitter_offsets.items():
                                dst_demo.attrs[f"augmentation_camera_jitter_{cam_name}_dx"] = float(cam_offset[0])
                                dst_demo.attrs[f"augmentation_camera_jitter_{cam_name}_dy"] = float(cam_offset[1])
                                dst_demo.attrs[f"augmentation_camera_jitter_{cam_name}_dz"] = float(cam_offset[2])

                        excluded_paths = ds.hdf5_excluded_augmented_paths()
                        _copy_group_tree(src_demo, dst_demo, excluded_paths)

                        dst_demo.create_dataset(ds.ACTIONS, data=executed_actions, dtype=np.float32)

                        if ds.OBS in dst_demo:
                            dst_obs = dst_demo[ds.OBS]
                        else:
                            dst_obs = dst_demo.create_group(ds.OBS)

                        _create_rgb_dataset(dst_obs, ds.OBS_RGB_TOP, rgb_top)
                        if ds.obs_path(ds.OBS_RGB_MAIN_45DEG) in src_demo:
                            _create_rgb_dataset(dst_obs, ds.OBS_RGB_MAIN_45DEG, rgb_main)
                        elif ds.obs_path(ds.OBS_RGB_MAIN) in src_demo:
                            _create_rgb_dataset(dst_obs, ds.OBS_RGB_MAIN, rgb_main)
                        else:
                            _create_rgb_dataset(dst_obs, ds.OBS_RGB_MAIN_45DEG, rgb_main)
                        _create_rgb_dataset(dst_obs, ds.OBS_RGB_WRIST, rgb_wrist)
                        _write_vision_freshness_datasets(
                            dst_obs,
                            num_steps=int(executed_actions.shape[0]),
                            vision_decimation=vision_decimation,
                        )
                        _write_recomputed_state_datasets(dst_demo, state_payload=state_payload)

                        print(
                            f"[INFO] Wrote {out_demo_name} <- {src_demo_name} "
                            f"variant={variant_type}, light={light_scale:.4f}, speed={speed_scale:.4f}, "
                            f"camera_jitter={jitter_name} (lights affected={light_count})."
                        )

            print(
                f"[INFO] Augmentation completed. Total output demos: {out_demo_index}, "
                f"kept_variants={kept_variant_count}, dropped_variants={dropped_variant_count}"
            )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as exc:
        print(f"\n[ERROR] Offline light augmentation failed: {exc}")
        print(f"[INFO] Closing simulation app (timeout={float(args_cli.close_timeout_sec):.1f}s)...")
        _close_simulation_app_with_timeout(
            simulation_app,
            timeout_sec=float(args_cli.close_timeout_sec),
            force_exit_code=1,
        )
        raise

    print(f"[INFO] Closing simulation app (timeout={float(args_cli.close_timeout_sec):.1f}s)...")
    _close_simulation_app_with_timeout(
        simulation_app,
        timeout_sec=float(args_cli.close_timeout_sec),
        force_exit_code=0,
    )


if __name__ == "__main__":
    main()
