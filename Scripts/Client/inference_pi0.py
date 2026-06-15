from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zmq

from isaaclab.app import AppLauncher

SERVER_DIR = Path(__file__).resolve().parent.parent / "Server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from isaaclab_env_module import (
    apply_camera_launch_workarounds,
    create_franka_simulation_session,
    reset_episode_state,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from task_registry import DEFAULT_TASK_ID, get_task_preset, get_task_timeout_seconds, list_task_presets
from task_eval_logging import format_task_eval_step_logs, get_eval_progress_value, should_print_eval_step
from task_success_evaluator import build_task_evaluator
from open_lid_evaluation import build_open_lid_drive_eval_result
from virtual_button_lid_opener import VirtualButtonLidOpener, VirtualButtonLidOpenerConfig
import policy_io as pio
from inference_protocol import PolicyServerError
from zmq_policy_client import RecoveringPolicyClient


def build_args():
    parser = argparse.ArgumentParser(description="IsaacLab client for PI0 remote policy inference")
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument("--task-id", type=str, default=DEFAULT_TASK_ID, help="Task preset id from task_registry")
    parser.add_argument("--list-tasks", action="store_true", help="List all available task presets and exit")
    parser.add_argument("--server-endpoint", type=str, default="tcp://127.0.0.1:5555", help="ZMQ REP server endpoint")
    parser.add_argument("--request-timeout-ms", type=int, default=300000, help="ZMQ request timeout in milliseconds")
    parser.add_argument(
        "--max-stale-actions",
        type=int,
        default=20,
        help="Max consecutive policy timeouts/errors before entering safe hold action",
    )
    parser.add_argument("--gripper-open-pos", type=float, default=0.04, help="Franka finger open target")
    parser.add_argument("--gripper-open-threshold", type=float, default=0.55, help="Open gripper threshold")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.45, help="Close gripper threshold")
    parser.add_argument(
        "--language-feature-mode",
        type=str,
        default="both",
        choices=["task", "language_instruction", "both"],
        help="Language input key mode for request payload.",
    )
    parser.add_argument("--top-feature-key", type=str, default="observation.images.top", help="Policy top camera feature key")
    parser.add_argument("--main-feature-key", type=str, default="observation.images.main", help="Policy main camera feature key")
    parser.add_argument("--wrist-feature-key", type=str, default="observation.images.wrist", help="Policy wrist camera feature key")
    parser.add_argument("--state-feature-key", type=str, default="observation.state", help="Policy state feature key")
    parser.add_argument(
        "--policy-hz",
        type=float,
        default=10,
        help="Override policy query rate. Default uses task_registry vision_hz.",
    )
    parser.add_argument(
        "--disable-fabric",
        "--disable_fabric",
        dest="disable_fabric",
        action="store_true",
        help="Disable IsaacLab fabric for debugging articulation/camera synchronization issues.",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write evaluation metrics.")
    parser.add_argument("--warmup-steps-after-reset", type=int, default=100, help="Warmup sim steps after each evaluation reset.")
    parser.add_argument(
        "--state-source",
        choices=["robot", "home", "commanded"],
        default="commanded",
        help=(
            "Source for observation.state. robot reads IsaacLab articulation state; "
            "home uses the task preset home state; commanded uses the last policy target. "
            "commanded avoids robot.data.joint_pos GPU->CPU stalls seen with PI0 online inference."
        ),
    )
    parser.add_argument("--z-print-interval-steps", type=int, default=30, help="Print current_z every N control steps in evaluation mode.")
    parser.set_defaults(enforce_root_pose_reset=False)
    parser.add_argument(
        "--enforce-root-pose-reset",
        dest="enforce_root_pose_reset",
        action="store_true",
        help="Force writing robot root pose on evaluation reset. Disabled by default for fixed-base Franka.",
    )
    parser.add_argument(
        "--no-enforce-root-pose-reset",
        dest="enforce_root_pose_reset",
        action="store_false",
        help="Do not force-write robot root pose on evaluation reset.",
    )
    return parser.parse_args()


def main():
    args_cli = build_args()

    if args_cli.list_tasks:
        print("[INFO] Available task presets:")
        for preset in list_task_presets():
            print(f"  - {preset.task_id}: {preset.description}")
        return

    task_preset = get_task_preset(args_cli.task_id)
    home_robot_state = np.array(
        [
            float(task_preset.robot_init_joint_pos.get("panda_joint1", 0.0)),
            float(task_preset.robot_init_joint_pos.get("panda_joint2", -0.785)),
            float(task_preset.robot_init_joint_pos.get("panda_joint3", 0.0)),
            float(task_preset.robot_init_joint_pos.get("panda_joint4", -2.356)),
            float(task_preset.robot_init_joint_pos.get("panda_joint5", 0.0)),
            float(task_preset.robot_init_joint_pos.get("panda_joint6", 1.571)),
            float(task_preset.robot_init_joint_pos.get("panda_joint7", 0.785)),
            float(task_preset.robot_init_joint_pos.get("panda_finger_joint.*", 0.04)),
        ],
        dtype=np.float32,
    )
    episodes = int(args_cli.episodes)
    if episodes <= 0:
        raise ValueError("PI0 inference is evaluation-only. Set --episodes to a positive integer.")

    args_cli = apply_camera_launch_workarounds(args_cli)
    if not hasattr(args_cli, "disable_fabric"):
        setattr(args_cli, "disable_fabric", False)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    camera_names = {spec.name for spec in task_preset.camera_specs}
    if "top" not in camera_names or "main" not in camera_names or "wrist" not in camera_names:
        raise ValueError(
            f"Task preset '{task_preset.task_id}' must define 'top', 'main', and 'wrist' cameras. "
            f"Current cameras: {sorted(camera_names)}"
        )

    session = create_franka_simulation_session(
        task_preset,
        sim_dt=1.0 / 60.0,
        render_interval=2,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
        reset_robot_root_pose=False,
        find_ee_body=False,
    )
    env_module = session.env_module
    arm_joint_ids = session.arm_joint_ids
    gripper_joint_ids = session.gripper_joint_ids

    policy_client = RecoveringPolicyClient(args_cli.server_endpoint, args_cli.request_timeout_ms)

    control_hz = max(1e-6, float(task_preset.control_hz))
    policy_hz = max(1e-6, float(args_cli.policy_hz if args_cli.policy_hz is not None else task_preset.vision_hz))
    if policy_hz > control_hz:
        print(f"[WARN] policy_hz ({policy_hz}) > control_hz ({control_hz}). Clamp policy_hz to control_hz.")
        policy_hz = control_hz

    policy_decimation = max(1, int(round(control_hz / policy_hz)))
    dt_target = 1.0 / control_hz

    print(f"[INFO] Using task preset: {task_preset.task_id}")
    print(f"[INFO] Scene USD: {task_preset.usd_path}")
    print(f"[INFO] Cameras from task_registry: {sorted(camera_names)}")
    print(f"[INFO] Connected to server: {args_cli.server_endpoint}")
    print("[INFO] Policy type: pi0")
    print(f"[INFO] Language feature mode: {args_cli.language_feature_mode}")
    print(f"[INFO] Policy loop: query every {policy_decimation} control steps (~{policy_hz:.2f}Hz)")

    centrifuge_eval = task_preset.centrifuge_eval
    pipette_eval = task_preset.pipette_eval
    timeout_seconds = float(get_task_timeout_seconds(task_preset))
    task_lower = str(task_preset.task_id).lower()
    is_open_lid_drive_task = task_lower == "open_the_centrifuge_lid"
    is_tube_task = task_lower == "pick_up_the_tube"
    is_pipette_task = "pipette" in task_lower
    is_centrifuge_task = task_lower == "close_the_centrifuge_lid"
    is_place_tube_on_balance_task = task_lower == "place_the_centrifuge_tube_on_the_balance"
    is_take_out_petri_task = task_lower == "take_out_the_petri_dish"
    is_place_petri_task = task_lower == "place_the_petri_dish"
    evaluator = None if is_open_lid_drive_task else build_task_evaluator(
        task_preset=task_preset,
        device=env_module.sim.device,
    )
    virtual_button_lid_opener = None
    gripper_probe_body_ids: list[int] = []
    if is_open_lid_drive_task:
        virtual_button_lid_opener = VirtualButtonLidOpener(
            VirtualButtonLidOpenerConfig(
                button_prim_path=centrifuge_eval.virtual_button_prim_path,
                joint_prim_path=centrifuge_eval.lid_joint_prim_path,
            ),
            device=env_module.sim.device,
        )
        virtual_button_lid_opener.initialize()
        gripper_probe_body_ids = pio.find_gripper_probe_body_ids(env_module.robot)

    print("[INFO] PI0 evaluation mode enabled.")
    eval_name = "open_lid_drive_target_position" if is_open_lid_drive_task else getattr(evaluator, "name", type(evaluator).__name__)
    print(f"[INFO] Evaluator: {eval_name}")
    print(f"[INFO] use_fabric={bool(getattr(env_module.sim.cfg, 'use_fabric', True))}")
    print(f"[INFO] Enforce robot root pose on reset: {bool(args_cli.enforce_root_pose_reset)}")
    print(f"[INFO] Episodes: {episodes}")
    if is_open_lid_drive_task:
        print(
            f"[INFO] Success rule from Enhance.py drive-target logic; timeout={timeout_seconds:.2f}s, "
            f"virtual button triggered and min(TargetPosition) < "
            f"{centrifuge_eval.drive_target_success_threshold_deg:.1f}deg."
        )
    elif is_tube_task:
        tube_eval = task_preset.tube_eval
        print(
            f"[INFO] Success rule from task_success_evaluator.py; "
            f"timeout={timeout_seconds:.2f}s, height_delta={tube_eval.success_height_delta:.4f}, "
            f"hold={tube_eval.success_hold_seconds:.2f}s, tilt_fail={tube_eval.tilt_fail_deg:.1f}deg"
        )
    elif is_pipette_task:
        print(
            f"[INFO] Success rule from task_success_evaluator.py; "
            f"timeout={timeout_seconds:.2f}s, pipette XY distance <= "
            f"{pipette_eval.success_xy_distance:.3f}m from Petri, "
            f"low-z fail threshold={pipette_eval.low_z_threshold:.3f}m."
        )
    elif is_centrifuge_task:
        comparator = ">" if str(centrifuge_eval.success_direction).lower() == "greater" else "<"
        print(
            f"[INFO] Success rule from task_success_evaluator.py; timeout={timeout_seconds:.2f}s, "
            f"{centrifuge_eval.lid_prim_path} X {comparator} "
            f"{centrifuge_eval.success_x_threshold_deg:.1f}deg for "
            f"{centrifuge_eval.success_hold_seconds:.2f}s."
        )
    elif is_take_out_petri_task:
        print(
            f"[INFO] Success rule from task_success_evaluator.py; timeout={timeout_seconds:.2f}s, "
            f"{task_preset.petri_close_prim_path} X < {task_preset.petri_success_x_threshold:.3f}, "
            f"Z > {task_preset.petri_success_z_threshold:.3f} for "
            f"{task_preset.petri_success_hold_seconds:.2f}s."
        )
    elif is_place_petri_task:
        print(
            f"[INFO] Success rule from task_success_evaluator.py; timeout={timeout_seconds:.2f}s, "
            f"{task_preset.petri_close_prim_path} Y < {task_preset.place_petri_success_y_max:.3f}, "
            f"X > {task_preset.place_petri_success_x_min:.3f}, "
            f"Z > {task_preset.place_petri_success_z_threshold:.3f}; "
            f"Z stable for {task_preset.place_petri_z_stable_seconds:.2f}s "
            f"(tol={task_preset.place_petri_z_stable_tolerance:.4f}m); "
            f"low-z fail threshold={task_preset.place_petri_success_z_threshold:.3f}m."
        )
    elif is_place_tube_on_balance_task:
        balance_eval = task_preset.place_tube_on_balance_eval
        print(
            f"[INFO] Success rule from task_success_evaluator.py; timeout={timeout_seconds:.2f}s, "
            f"{task_preset.tube_prim_path} XY inside {balance_eval.plate_prim_path} circle "
            f"(radius_scale={balance_eval.plate_radius_scale:.2f}), "
            f"{balance_eval.tube_z_min:.3f} < Z < {balance_eval.tube_z_max:.3f} for "
            f"{balance_eval.success_hold_seconds:.2f}s."
        )
    else:
        print(f"[INFO] Success rule from task_success_evaluator.py; timeout={timeout_seconds:.2f}s")

    success_count = 0
    fail_count = 0
    episode_details: list[dict] = []
    feature_keys = pio.PolicyFeatureKeys(
        top=args_cli.top_feature_key,
        main=args_cli.main_feature_key,
        wrist=args_cli.wrist_feature_key,
        state=args_cli.state_feature_key,
    )

    try:
        for ep_idx in range(1, episodes + 1):
            if virtual_button_lid_opener is not None:
                if ep_idx == 1:
                    virtual_button_lid_opener.reset()
                else:
                    virtual_button_lid_opener.reset_to_default_closed_state()
            if not (is_open_lid_drive_task and ep_idx == 1):
                reset_episode_state(
                    env_module,
                    warmup_steps=0 if is_open_lid_drive_task else int(args_cli.warmup_steps_after_reset),
                    enforce_root_pose_reset=bool(args_cli.enforce_root_pose_reset),
                )

            curr_target = pio.build_safe_hold_action(env_module.robot, arm_joint_ids, env_module.sim.device)
            action_chunk = curr_target.view(1, -1)
            commanded_state_np = home_robot_state.copy()
            action_chunk_np = commanded_state_np.reshape(1, -1).copy()
            chunk_cursor = 0
            control_step_count = 0
            consecutive_comm_failures = 0
            request_policy_reset = True
            gripper_open_state = True
            stale_z_steps = 0
            last_printed_z = None
            episode_elapsed = 0.0
            outcome = "timeout_fail"
            outcome_reason = "timeout"
            open_lid_policy_hold_active = False

            drive_target_initial = None
            drive_target_min = None
            drive_target_max = None
            drive_target_current = None
            if is_open_lid_drive_task and virtual_button_lid_opener is not None:
                drive_target_initial = virtual_button_lid_opener.get_drive_target_position()
                drive_target_current = drive_target_initial
                drive_target_min = drive_target_initial
                drive_target_max = drive_target_initial
                eval_result = build_open_lid_drive_eval_result(
                    initial_target_deg=drive_target_initial,
                    min_target_deg=drive_target_min,
                    max_target_deg=drive_target_max,
                    triggered=bool(virtual_button_lid_opener.has_triggered),
                    success_threshold_deg=float(centrifuge_eval.drive_target_success_threshold_deg),
                )
            else:
                evaluator.reset_episode()
                eval_result = evaluator.finalize()
            if eval_result.reason == "tube_pose_unavailable":
                raise RuntimeError(f"Cannot read tube pose from prim: {task_preset.tube_prim_path}")
            if eval_result.reason == "tube_or_plate_pose_unavailable":
                raise RuntimeError(
                    f"Cannot read tube or balance plate pose from prims: "
                    f"{task_preset.tube_prim_path}, {task_preset.place_tube_on_balance_eval.plate_prim_path}"
                )
            if eval_result.reason == "petri_close_pose_unavailable":
                raise RuntimeError(f"Cannot read Petri_close pose from prim: {task_preset.petri_close_prim_path}")
            if eval_result.reason == "water_bath_lid_pose_unavailable":
                raise RuntimeError(f"Cannot read water bath lid pose from prim: {task_preset.water_bath_lid_eval.lid_prim_path}")
            if eval_result.reason == "centrifuge_lid_pose_unavailable":
                raise RuntimeError(f"Cannot read centrifuge lid pose from prim: {centrifuge_eval.lid_prim_path}")
            if eval_result.reason == "spectrophotometer_lid_pose_unavailable":
                raise RuntimeError(f"Cannot read spectrophotometer lid pose from prim: {centrifuge_eval.lid_prim_path}")
            if eval_result.reason == "open_lid_drive_target_unavailable":
                raise RuntimeError("Cannot read centrifuge lid drive TargetPosition.")

            dt = float(env_module.sim.cfg.dt)
            sim_substeps = max(1, int(round((1.0 / control_hz) / dt)))
            episode_total_steps = max(1, int(math.ceil(timeout_seconds * control_hz)))

            while simulation_app.is_running():
                tick_start = time.perf_counter()
                control_step_count += 1

                rgb_top = env_module.capture_rgb("top")
                rgb_main = env_module.capture_rgb("main")
                rgb_wrist = env_module.capture_rgb("wrist")
                top_chw = pio.rgb_to_uint8_chw_numpy(rgb_top)
                main_chw = pio.rgb_to_uint8_chw_numpy(rgb_main)
                wrist_chw = pio.rgb_to_uint8_chw_numpy(rgb_wrist)

                if top_chw is None or main_chw is None or wrist_chw is None:
                    env_module.sim.step(render=True)
                    env_module.robot.update(env_module.sim.cfg.dt)
                    continue

                if args_cli.state_source == "home":
                    robot_state = home_robot_state.copy()
                elif args_cli.state_source == "commanded":
                    robot_state = commanded_state_np.copy()
                else:
                    env_module.robot.update(env_module.sim.cfg.dt)
                    joint_pos = env_module.robot.data.joint_pos
                    arm_state = joint_pos[:, arm_joint_ids][0].detach().to(dtype=torch.float32)
                    gripper_state = joint_pos[:, gripper_joint_ids[0:1]][0].detach().to(dtype=torch.float32)
                    robot_state = (
                        torch.cat([arm_state, gripper_state], dim=0)
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .numpy()
                        .astype(np.float32, copy=False)
                    )

                should_query_policy = ((control_step_count - 1) % policy_decimation == 0)
                if should_query_policy:
                    if (
                        is_open_lid_drive_task
                        and virtual_button_lid_opener is not None
                        and virtual_button_lid_opener.has_triggered
                    ):
                        curr_target = pio.build_safe_hold_action(
                            env_module.robot,
                            arm_joint_ids,
                            env_module.sim.device,
                        )
                        action_chunk = curr_target.view(1, -1)
                        action_chunk_np = (
                            curr_target.detach()
                            .to(device="cpu", dtype=torch.float32)
                            .numpy()
                            .reshape(1, -1)
                            .copy()
                        )
                        commanded_state_np = action_chunk_np[0, : home_robot_state.shape[0]].copy()
                        chunk_cursor = 0
                        consecutive_comm_failures = 0
                        request_policy_reset = False
                        if not open_lid_policy_hold_active:
                            print(
                                f"[EP {ep_idx:03d}] open lid trigger detected; "
                                "hold robot and continue lid drive evaluation."
                            )
                            open_lid_policy_hold_active = True
                    else:
                        obs_frame = pio.build_request_observation(
                            top_rgb=top_chw,
                            main_rgb=main_chw,
                            wrist_rgb=wrist_chw,
                            robot_state=robot_state,
                            task=task_preset.language_instruction,
                            reset_policy=request_policy_reset,
                            feature_keys=feature_keys,
                            language_feature_mode=args_cli.language_feature_mode,
                        )
                        try:
                            action_list = policy_client.request_action(obs_frame)
                            action_tensor = torch.tensor(action_list, dtype=torch.float32, device=env_module.sim.device)
                            action_1d, action_chunk, chunk_cursor = pio.pick_action_from_chunk(action_tensor)
                            action_np = np.asarray(action_list, dtype=np.float32)
                            if action_np.ndim == 2 and action_np.shape[0] > 0:
                                action_chunk_np = action_np.copy()
                                commanded_state_np = action_np[0, : home_robot_state.shape[0]].copy()
                            elif action_np.ndim == 1:
                                action_chunk_np = action_np.reshape(1, -1).copy()
                                commanded_state_np = action_np[: home_robot_state.shape[0]].copy()
                            curr_target = action_1d
                            consecutive_comm_failures = 0
                            request_policy_reset = False
                        except PolicyServerError as exc:
                            consecutive_comm_failures += 1
                            curr_target = pio.build_safe_hold_action(
                                env_module.robot,
                                arm_joint_ids,
                                env_module.sim.device,
                            )
                            action_chunk = curr_target.view(1, -1)
                            action_chunk_np = (
                                curr_target.detach()
                                .to(device="cpu", dtype=torch.float32)
                                .numpy()
                                .reshape(1, -1)
                                .copy()
                            )
                            commanded_state_np = action_chunk_np[0, : home_robot_state.shape[0]].copy()
                            chunk_cursor = 0
                            request_policy_reset = True
                            print(
                                f"[ERROR] Policy server rejected inference "
                                f"({exc.error_type}: {exc}). Enter safe hold immediately."
                            )
                        except zmq.error.Again:
                            consecutive_comm_failures += 1
                            print("[WARN] ZMQ timeout. Socket rebuilt; keep current policy target/chunk.")
                        except Exception as exc:
                            consecutive_comm_failures += 1
                            print(f"[WARN] ZMQ communication failed: {exc}. Keep current policy target/chunk.")

                        if consecutive_comm_failures >= max(1, int(args_cli.max_stale_actions)):
                            curr_target = pio.build_safe_hold_action(
                                env_module.robot,
                                arm_joint_ids,
                                env_module.sim.device,
                            )
                            action_chunk = curr_target.view(1, -1)
                            chunk_cursor = 0
                            request_policy_reset = True
                elif action_chunk.ndim == 2 and (chunk_cursor + 1) < int(action_chunk.shape[0]):
                    chunk_cursor += 1
                    curr_target = action_chunk[chunk_cursor]
                    if (
                        args_cli.state_source == "commanded"
                        and action_chunk_np.ndim == 2
                        and chunk_cursor < action_chunk_np.shape[0]
                    ):
                        commanded_state_np = action_chunk_np[chunk_cursor, : home_robot_state.shape[0]].copy()

                gripper_open_state = pio.apply_franka_action_hysteresis(
                    robot=env_module.robot,
                    action_1d=curr_target,
                    arm_joint_ids=arm_joint_ids,
                    gripper_joint_ids=gripper_joint_ids,
                    gripper_open_pos=args_cli.gripper_open_pos,
                    prev_gripper_open=gripper_open_state,
                    gripper_open_threshold=float(args_cli.gripper_open_threshold),
                    gripper_close_threshold=float(args_cli.gripper_close_threshold),
                )

                for sub_idx in range(sim_substeps):
                    env_module.sim.step(render=(sub_idx == sim_substeps - 1))
                    env_module.robot.update(env_module.sim.cfg.dt)
                    if virtual_button_lid_opener is not None:
                        probe_positions = pio.get_gripper_probe_positions(env_module.robot, gripper_probe_body_ids)
                        if probe_positions is not None:
                            virtual_button_lid_opener.update(probe_positions, dt)
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

                if is_open_lid_drive_task:
                    eval_result = build_open_lid_drive_eval_result(
                        initial_target_deg=drive_target_initial,
                        min_target_deg=drive_target_min,
                        max_target_deg=drive_target_max,
                        triggered=bool(
                            virtual_button_lid_opener is not None
                            and virtual_button_lid_opener.has_triggered
                        ),
                        success_threshold_deg=float(centrifuge_eval.drive_target_success_threshold_deg),
                    )
                else:
                    evaluator.update(dt * sim_substeps)
                    eval_result = evaluator.finalize()
                eval_metrics = dict(eval_result.metrics)
                current_z = get_eval_progress_value(eval_result)

                if should_print_eval_step(control_step_count, int(args_cli.z_print_interval_steps)):
                    for log_line in format_task_eval_step_logs(
                        task_preset=task_preset,
                        eval_result=eval_result,
                        evaluator=evaluator,
                        step_index=control_step_count,
                        total_steps=episode_total_steps,
                        episode_index=ep_idx,
                        drive_target_current_deg=drive_target_current,
                    ):
                        print(log_line)

                if not math.isfinite(float(current_z)):
                    stale_z_steps = 0
                elif last_printed_z is None or math.fabs(float(current_z) - last_printed_z) > 1.0e-5:
                    stale_z_steps = 0
                else:
                    stale_z_steps += 1
                if math.isfinite(float(current_z)):
                    last_printed_z = float(current_z)

                if stale_z_steps == 180:
                    print(
                        "[WARN] current_z appears unchanged for a long horizon. "
                        "Check task_success_evaluator.py pose backend metrics in the JSON report."
                    )

                if eval_result.reason == "tube_pose_unavailable":
                    outcome = "invalid_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.reason == "tube_or_plate_pose_unavailable":
                    outcome = "invalid_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.reason == "tube_tilt_fail":
                    outcome = "tilt_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.reason == "pipette_z_below_threshold":
                    outcome = "low_z_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.reason == "petri_close_pose_unavailable":
                    outcome = "invalid_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.reason == "water_bath_lid_pose_unavailable":
                    outcome = "invalid_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.reason == "centrifuge_lid_pose_unavailable":
                    outcome = "invalid_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.reason == "spectrophotometer_lid_pose_unavailable":
                    outcome = "invalid_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.reason == "open_lid_drive_target_unavailable":
                    outcome = "invalid_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.reason == "petri_close_z_below_threshold":
                    outcome = "low_z_fail"
                    outcome_reason = eval_result.reason
                    break
                if eval_result.success:
                    outcome = "success"
                    outcome_reason = eval_result.reason
                    break

                episode_elapsed += dt * sim_substeps
                if episode_elapsed >= timeout_seconds:
                    outcome = "timeout_fail"
                    outcome_reason = f"elapsed={episode_elapsed:.2f}s"
                    break

                elapsed = time.perf_counter() - tick_start
                sleep_s = dt_target - elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

            success = outcome == "success"
            success_count += int(success)
            fail_count += int(not success)
            if is_open_lid_drive_task:
                final_metrics = dict(eval_result.metrics)
            else:
                final_metrics = dict(evaluator.finalize().metrics)
            episode_details.append(
                {
                    "episode": ep_idx,
                    "success": bool(success),
                    "outcome": outcome,
                    "reason": outcome_reason,
                    "metrics": final_metrics,
                }
            )
            print(f"[EP {ep_idx:03d}] {outcome} ({outcome_reason})")

        total = success_count + fail_count
        success_rate = (success_count / total) if total > 0 else 0.0
        print("\n========== PI0 Evaluation Summary ==========")
        print(f"episodes={total}")
        print(f"success={success_count}")
        print(f"failure={fail_count}")
        print(f"success_rate={success_rate:.4f}")
        print("============================================")

        if args_cli.output_json:
            output = {
                "task_id": task_preset.task_id,
                "execution_logic": "pi0_inference_top_main_hysteresis_gripper",
                "tube_prim_path": task_preset.tube_prim_path,
                "pipette_prim_path": task_preset.pipette_prim_path,
                "petri_prim_path": task_preset.petri_prim_path,
                "eval_type": eval_name,
                "episodes": total,
                "success": success_count,
                "failure": fail_count,
                "success_rate": success_rate,
                "policy_hz": float(policy_hz),
                "control_hz": float(control_hz),
                "timeout_seconds": timeout_seconds,
                "eval_config": {
                    "petri_close_prim_path": task_preset.petri_close_prim_path,
                    "pipette_success_xy_distance": float(pipette_eval.success_xy_distance),
                    "pipette_low_z_threshold": float(pipette_eval.low_z_threshold),
                    "petri_success_x_threshold": float(task_preset.petri_success_x_threshold),
                    "petri_success_z_threshold": float(task_preset.petri_success_z_threshold),
                    "petri_success_hold_seconds": float(task_preset.petri_success_hold_seconds),
                    "place_petri_success_x_min": float(task_preset.place_petri_success_x_min),
                    "place_petri_success_y_max": float(task_preset.place_petri_success_y_max),
                    "place_petri_success_z_threshold": float(task_preset.place_petri_success_z_threshold),
                    "place_petri_z_stable_seconds": float(task_preset.place_petri_z_stable_seconds),
                    "place_petri_z_stable_tolerance": float(task_preset.place_petri_z_stable_tolerance),
                    "centrifuge_lid_prim_path": centrifuge_eval.lid_prim_path,
                    "open_lid_eval_source": "drive_target_position",
                    "open_lid_drive_target_success_threshold_deg": float(
                        centrifuge_eval.drive_target_success_threshold_deg
                    ),
                    "place_tube_on_balance_plate_prim_path": task_preset.place_tube_on_balance_eval.plate_prim_path,
                    "place_tube_on_balance_plate_radius_scale": float(task_preset.place_tube_on_balance_eval.plate_radius_scale),
                    "place_tube_on_balance_z_min": float(task_preset.place_tube_on_balance_eval.tube_z_min),
                    "place_tube_on_balance_z_max": float(task_preset.place_tube_on_balance_eval.tube_z_max),
                    "place_tube_on_balance_success_hold_seconds": float(task_preset.place_tube_on_balance_eval.success_hold_seconds),
                },
                "details": episode_details,
            }
            output_path = Path(args_cli.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Wrote JSON report: {output_path}")
    finally:
        policy_client.close()
        simulation_app.close()
    return


if __name__ == "__main__":
    main()
