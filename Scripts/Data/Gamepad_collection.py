import argparse
import os
import sys
import time
from pathlib import Path
import torch
import h5py
from isaaclab.app import AppLauncher
import schema as ds

SERVER_DIR = Path(__file__).resolve().parents[1] / "Server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from isaaclab_env_module import (
    apply_camera_launch_workarounds,
    create_franka_simulation_session,
)
from task_registry import DEFAULT_TASK_ID, OPEN_CENTRIFUGE_LID_TASK_ID, get_task_preset, list_task_presets

# ── 1. CLI 参数与 App 启动 ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_demos", type=int, default=5)
    parser.add_argument("--task_id", type=str, default=DEFAULT_TASK_ID)
    parser.add_argument("--list_tasks", action="store_true", help="List all available task presets and exit.")
    parser.add_argument("--disable_virtual_button_open_lid", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    if args_cli.list_tasks:
        print("[INFO] Available task presets:")
        for preset in list_task_presets():
            print(f"  - {preset.task_id}: {preset.description}")
        raise SystemExit(0)

    task_preset = get_task_preset(args_cli.task_id)

    args_cli = apply_camera_launch_workarounds(args_cli)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # ── 2. 导入 Isaac Lab 模块 (必须在 App 启动后) ────────────────────
    import omni.usd
    import isaaclab.utils.math as math_utils
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    try:
        from isaaclab.devices import Se3Gamepad, Se3GamepadCfg
    except ImportError:
        from isaaclab.devices import Se3Gamepad
        Se3GamepadCfg = None
    import carb

    # IsaacLab 2.3 使用通用数据集接口导出 HDF5
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler


    class OfficialEpisodeCollector:
        """对齐官方 record_demos/replay_demos 数据组织的简化收集器。"""

        def __init__(self, dataset_file: str, env_name: str, num_demos: int = 0):
            self.dataset_file = dataset_file
            self.env_name = env_name
            self.num_demos = num_demos

            output_dir = os.path.dirname(self.dataset_file) or "."
            output_name = os.path.splitext(os.path.basename(self.dataset_file))[0]
            os.makedirs(output_dir, exist_ok=True)

            self._dataset_handler = HDF5DatasetFileHandler()
            # 数据集文件名保持不变：存在则在原文件后追加，不存在则创建新文件。
            dataset_stem_path = os.path.join(output_dir, output_name)
            dataset_hdf5_path = f"{dataset_stem_path}.hdf5"
            if os.path.exists(dataset_hdf5_path):
                # 官方 open() 是只读模式，不能直接用于 write_episode。
                # 这里改为 h5py append("a") 可写打开，并对齐 handler 内部游标。
                self._dataset_handler._hdf5_file_stream = h5py.File(dataset_hdf5_path, "a")
                self._dataset_handler._hdf5_data_group = self._dataset_handler._hdf5_file_stream.require_group(ds.DATA_GROUP)

                demo_ids: list[int] = []
                for name in self._dataset_handler._hdf5_data_group.keys():
                    index = ds.demo_sort_key(name)
                    if index < 10**12:
                        demo_ids.append(index)

                next_demo_id = (max(demo_ids) + 1) if demo_ids else 0
                self._dataset_handler._demo_count = next_demo_id
                self.existing_episode_count = len(demo_ids)

                try:
                    existing_env_name = self._dataset_handler.get_env_name()
                except Exception:
                    existing_env_name = None

                if existing_env_name is None:
                    # 兼容部分旧文件：如果缺 env_name 元数据则补上。
                    self._dataset_handler.set_env_name(self.env_name)
                    existing_env_name = self.env_name

                if existing_env_name != self.env_name:
                    print(
                        f"[WARN] Existing dataset env_name='{existing_env_name}' != requested env_name='{self.env_name}'. "
                        "Appending anyway."
                    )

                print(
                    f"[INFO] Append mode enabled: {dataset_hdf5_path} "
                    f"(existing episodes: {self.existing_episode_count}, next demo id: {next_demo_id})"
                )
            else:
                # create 支持传入不带扩展名的路径，行为与官方脚本一致。
                self._dataset_handler.create(dataset_stem_path, env_name=self.env_name)
                self.existing_episode_count = 0
                print(f"[INFO] Create mode enabled: {dataset_hdf5_path}")

            self._episode = EpisodeData()
            self.exported_successful_episode_count = 0
            self.exported_failed_episode_count = 0

        def _prepare_episode_for_export(self):
            # 不同 IsaacLab 版本对 EpisodeData 的导出预处理 API 存在差异。
            if hasattr(self._episode, "pre_export") and callable(self._episode.pre_export):
                self._episode.pre_export()
                return

            # 兼容旧实现：手动把嵌套字典中的 list[Tensor] 堆叠成 Tensor。
            def _stack_leaf_lists(node):
                for key, value in node.items():
                    if isinstance(value, list):
                        if len(value) > 0 and torch.is_tensor(value[0]):
                            node[key] = torch.stack(value)
                    elif isinstance(value, dict):
                        _stack_leaf_lists(value)

            _stack_leaf_lists(self._episode.data)

        def reset_episode(self):
            self._episode = EpisodeData()

        def has_data(self) -> bool:
            return not self._episode.is_empty()

        def set_initial_state(self, initial_state: dict[str, torch.Tensor]):
            for key, value in initial_state.items():
                self._episode.add(ds.initial_path(key), value.detach().clone())

        def add_step(
            self,
            obs_dict: dict[str, torch.Tensor],
            actions: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor,
            state_dict: dict[str, torch.Tensor] | None = None,
        ):
            self._episode.add(ds.ACTIONS, actions.detach().clone())

            for key, value in obs_dict.items():
                self._episode.add(ds.obs_path(key), value.detach().clone())

            if state_dict is not None:
                for key, value in state_dict.items():
                    self._episode.add(ds.state_path(key), value.detach().clone())

            self._episode.add("rewards", reward.detach().clone())
            self._episode.add("dones", done.detach().clone())

        def export_episode(self, success: bool) -> bool:
            if self._episode.is_empty():
                return False

            self._episode.success = success
            self._prepare_episode_for_export()
            self._dataset_handler.write_episode(self._episode)
            self._dataset_handler.flush()

            if success:
                self.exported_successful_episode_count += 1
            else:
                self.exported_failed_episode_count += 1

            self.reset_episode()
            return True

        def close(self):
            self._dataset_handler.close()

    # 配置参数
    _SENS = float(task_preset.sensitivity)
    POS_SENSITIVITY = 0.002 * _SENS
    ROT_SENSITIVITY = 0.01 * _SENS
    CAMERA_WIDTH = max(32, int(task_preset.camera_width))
    CAMERA_HEIGHT = max(32, int(task_preset.camera_height))
    CAMERA_PATHS = {spec.name: spec.prim_path for spec in task_preset.camera_specs}
    for required_camera_name in ("main", "top", "wrist"):
        if required_camera_name not in CAMERA_PATHS:
            raise ValueError(
                f"Task preset '{task_preset.task_id}' misses required camera '{required_camera_name}'."
            )
    MAIN_CAM_PATH = CAMERA_PATHS["main"]
    TOP_CAM_PATH = CAMERA_PATHS["top"]
    WRIST_CAM_PATH = CAMERA_PATHS["wrist"]

    print(f"[INFO] Selected task preset: {task_preset.task_id}")

    # ── 3. 场景与 Franka 初始化 ───────────────────
    session = create_franka_simulation_session(
        task_preset,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_sensor_type=task_preset.camera_sensor_type.lower(),
        warmup_render_steps=6,
        sim_dt=1 / 60.0,
        render_interval=4,
        create_camera_prims=False,
        create_sensor_cameras=False,
        find_ee_body=True,
    )
    env_module = session.env_module
    sim = session.sim
    robot = session.robot


    # 获取设备信息 (通常是 'cuda:0')
    device = sim.device

    # ── 5. 控制器设置 (DifferentialIKController) ───────────────────────
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        ik_method="dls",      # 阻尼最小二乘法，适合遥操
        ik_params={
            "lambda_val": 0.05  # 将阻尼系数放在这里
        }
    )
    # 注意：num_envs=1 是因为你只有一个机器人
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=device)

    # 确定需要控制的部位
    # 机械臂的关节索引（不包含夹爪）
    arm_joint_ids = session.arm_joint_ids
    # 夹爪的关节索引
    gripper_joint_ids = session.gripper_joint_ids
    # 末端执行器的 Body 索引
    ee_body_id = session.ee_body_id
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

    # ── 6. 手柄遥操设置 ────────────────────────────────────────────────
    if Se3GamepadCfg is not None:
        teleop_interface = Se3Gamepad(
            Se3GamepadCfg(pos_sensitivity=POS_SENSITIVITY, rot_sensitivity=ROT_SENSITIVITY)
        )
    else:
        teleop_interface = Se3Gamepad(
            pos_sensitivity=POS_SENSITIVITY,
            rot_sensitivity=ROT_SENSITIVITY,
        )

    GAMEPAD_COMMAND_DEADZONE = 1.0e-4

    should_reset = False
    should_skip = False
    should_exit = False


    def reset_recording_instance():
        nonlocal should_reset
        should_reset = True


    def skip_recording_instance():
        nonlocal should_skip
        should_skip = True


    def request_exit():
        nonlocal should_exit
        should_exit = True
        print("\n[INFO] Exit requested by gamepad.")

    def gamepad_input(name: str):
        gamepad_enum = getattr(carb.input, "GamepadInput", None)
        if gamepad_enum is None:
            return name
        return getattr(gamepad_enum, name, name)


    def normalize_gamepad_output(gamepad_output):
        gripper_cmd = False

        if isinstance(gamepad_output, tuple):
            gamepad_delta_pose, gamepad_gripper_cmd = gamepad_output
            gamepad_command = torch.as_tensor(gamepad_delta_pose, dtype=torch.float32, device=device).flatten()
            gripper_cmd = bool(gamepad_gripper_cmd)
        else:
            gamepad_command = torch.as_tensor(gamepad_output, dtype=torch.float32, device=device).flatten()
            if gamepad_command.numel() >= 7:
                gripper_cmd = bool(gamepad_command[6] > 0.0)

        if gamepad_command.numel() < 6:
            raise RuntimeError(
                f"Se3Gamepad.advance() returned {gamepad_command.numel()} values; expected at least 6."
            )

        if gamepad_command.numel() < 7:
            gamepad_command = torch.cat([gamepad_command[:6], torch.zeros(1, dtype=torch.float32, device=device)])
        else:
            gamepad_command = gamepad_command[:7]

        gamepad_command[:6] = torch.where(
            gamepad_command[:6].abs() < GAMEPAD_COMMAND_DEADZONE,
            torch.zeros_like(gamepad_command[:6]),
            gamepad_command[:6],
        )
        if not gripper_cmd:
            gripper_cmd = bool(gamepad_command[6] > 0.0)

        return gamepad_command, gripper_cmd

    teleop_interface.add_callback(gamepad_input("A"), reset_recording_instance)
    teleop_interface.add_callback(gamepad_input("B"), skip_recording_instance)
    teleop_interface.add_callback(gamepad_input("Y"), request_exit)
    teleop_interface.reset()

    # ── 7. 初始化目标位姿 ──────────────────────────────────────────────
    # 从张量数据中获取初始的末端位姿 (Shape: [1, 3] 和 [1, 4])
    target_pos = robot.data.body_pos_w[:, ee_body_id].clone()
    target_quat = robot.data.body_quat_w[:, ee_body_id].clone()
    gripper_open_target = torch.full((1, 2), 0.04, dtype=torch.float32, device=device)
    gripper_close_target = torch.zeros((1, 2), dtype=torch.float32, device=device)

    print("\n=== Isaac Lab Teleoperation Ready ===")
    print('[INFO] Gamepad buttons: "A" save current demo and next, "B" skip current demo and next, "Y" exit.')
    print('[INFO] Default gamepad control: "X" toggles gripper; sticks move xyz/yaw; D-pad controls roll/pitch.')

    # ==================== 创建俯视相机与“自动嵌入式”分屏视窗 ====================
    import omni.kit.viewport.utility as vp_utils
    from pxr import Sdf
    import omni.ui as ui
    import omni.kit.app
    import asyncio
    from virtual_button_lid_opener import VirtualButtonLidOpener, VirtualButtonLidOpenerConfig


    def _set_viewport_resolution(viewport_window, width: int, height: int):
        if viewport_window is None:
            return
        vp_api = getattr(viewport_window, "viewport_api", None)
        if vp_api is None:
            return
        try:
            if hasattr(vp_api, "set_texture_resolution"):
                vp_api.set_texture_resolution((width, height))
            elif hasattr(vp_api, "resolution"):
                vp_api.resolution = (width, height)
        except Exception as e:
            win_title = getattr(viewport_window, "title", "unknown")
            print(f"[WARN] Failed to set viewport resolution for {win_title}: {e}")


    WRIST_CAMERA_REFRESH_NUDGE_M = 1.0e-5
    wrist_camera_refresh_phase = False


    def _set_viewport_camera_path(viewport_window, camera_path: str):
        if viewport_window is None:
            return
        vp_api = getattr(viewport_window, "viewport_api", viewport_window)
        if vp_api is None:
            return
        try:
            vp_api.camera_path = Sdf.Path(camera_path)
        except Exception as e:
            win_title = getattr(viewport_window, "title", "unknown")
            print(f"[WARN] Failed to set camera path for {win_title}: {e}")


    def refresh_wrist_camera_viewport(render_frames: int = 3):
        nonlocal wrist_camera_refresh_phase

        wrist_camera_refresh_phase = not wrist_camera_refresh_phase
        z_offset = -WRIST_CAMERA_REFRESH_NUDGE_M if wrist_camera_refresh_phase else 0.0

        try:
            env_module.refresh_camera_prim("wrist", translation_offset=(0.0, 0.0, z_offset))
        except Exception as e:
            print(f"[WARN] Failed to refresh wrist camera prim: {e}")
            return

        _set_viewport_camera_path(wrist_vp_window, WRIST_CAM_PATH)

        # Isaac Sim 5.1 can keep a stale viewport camera matrix for cameras parented
        # under an articulation after reset. A tiny authored xform change plus a few
        # render ticks matches the manual Property panel edit without visibly moving
        # the wrist camera.
        for _ in range(max(0, int(render_frames))):
            sim.render()


    # 1) 通过环境模块创建全局主俯视相机（45°）+ 顶视相机 + 挂载在 panda_hand 的腕部相机
    env_module.define_camera_prims()

    # 2) 创建新视口：左上腕部，左下顶视
    wrist_vp_window = vp_utils.create_viewport_window(
        "Wrist View",
        camera_path=WRIST_CAM_PATH,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )
    top_vp_window = vp_utils.create_viewport_window(
        "Top-Down View",
        camera_path=TOP_CAM_PATH,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
    )

    # 主视口切到主俯视相机
    main_vp = vp_utils.get_viewport_from_window_name("Viewport")
    if main_vp is not None:
        main_vp.camera_path = Sdf.Path(MAIN_CAM_PATH)

    _set_viewport_resolution(wrist_vp_window, CAMERA_WIDTH, CAMERA_HEIGHT)
    _set_viewport_resolution(top_vp_window, CAMERA_WIDTH, CAMERA_HEIGHT)

    # 3) 仅使用 IsaacLab 2.3 原生相机管线采集（Camera / TiledCamera）。
    sensor_cameras = {}
    sensor_type = task_preset.camera_sensor_type.lower()
    print(f"[INFO] RGB backend: isaaclab.{sensor_type}")
    try:
        sensor_cameras = env_module.create_sensor_cameras()
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize isaaclab {sensor_type} sensor pipeline: {e}. "
            "Please verify --enable_cameras/offscreen settings and camera prim path."
        ) from e

    for cam_name in ("main", "top", "wrist"):
        if cam_name in sensor_cameras:
            print(f"[INFO] Sensor camera ready: {cam_name} ({sensor_type})")

    refresh_wrist_camera_viewport()

    # 二次 reset 后刷新末端目标位姿，保持后续遥操与记录一致。
    target_pos = robot.data.body_pos_w[:, ee_body_id].clone()
    target_quat = robot.data.body_quat_w[:, ee_body_id].clone()

    # 4) 自动 UI 布局：右侧主俯视占 1/2，左上腕部，左下 top
    virtual_button_lid_opener = None
    if task_preset.task_id == OPEN_CENTRIFUGE_LID_TASK_ID and not args_cli.disable_virtual_button_open_lid:
        virtual_button_lid_opener = VirtualButtonLidOpener(
            VirtualButtonLidOpenerConfig(),
            device=device,
        )
        virtual_button_lid_opener.initialize()

    async def dock_window():
        # 必须等待一帧，确保底层的 UI 实例已经成功渲染出来
        await omni.kit.app.get_app().next_update_async()
        await omni.kit.app.get_app().next_update_async()
    
        # 获取主视窗和新视窗
        main_win = ui.Workspace.get_window("Viewport")
        wrist_win = ui.Workspace.get_window("Wrist View")
        top_win = ui.Workspace.get_window("Top-Down View")
    
        if main_win and wrist_win and top_win:
            # 先左右分栏: 主视口在右半边，wrist 在左半边
            wrist_win.dock_in(main_win, ui.DockPosition.LEFT, ratio=0.5)
            # 再在左栏上下切分: 左上 wrist, 左下 top
            top_win.dock_in(wrist_win, ui.DockPosition.BOTTOM, ratio=0.5)
            refresh_wrist_camera_viewport(render_frames=0)

    # 触发该异步排版任务
    dock_task = asyncio.ensure_future(dock_window())
    # =================================================================================

    # ==================== 新增：初始化数据收集器 ====================
    print("\n[INFO] Initializing Data Collector...")
    # 对齐官方 record_demos：通过 dataset_file 控制导出路径和文件名。
    collector = OfficialEpisodeCollector(
        dataset_file=task_preset.dataset_file,
        env_name=task_preset.env_name,
        num_demos=args_cli.num_demos,
    )

    instruction_text = (task_preset.language_instruction or "pick up the tube").strip()
    if not instruction_text:
        instruction_text = "pick up the tube"
    instruction_bytes = instruction_text.encode("utf-8")
    language_instruction_tensor = torch.tensor(
        list(instruction_bytes), dtype=torch.uint8, device=device
    ).unsqueeze(0)
    language_instruction_length = torch.tensor(
        [len(instruction_bytes)], dtype=torch.int32, device=device
    )

    # 缓存任务预设下的“正确”初始关节状态，避免后续 robot.reset() 回退到 USD authored 默认值。
    cached_init_joint_pos = robot.data.joint_pos.clone()
    cached_init_joint_vel = torch.zeros_like(robot.data.joint_vel)


    def capture_initial_state_for_episode():
        initial_state = {
            ds.INITIAL_ROBOT_ROOT_STATE: robot.data.root_state_w[:, :13].clone(),
            ds.INITIAL_ROBOT_JOINT_POS: robot.data.joint_pos.clone(),
            ds.INITIAL_ROBOT_JOINT_VEL: robot.data.joint_vel.clone(),
            "target_eef_pos": target_pos.clone(),
            "target_eef_quat": target_quat.clone(),
            ds.INITIAL_LANGUAGE_UTF8: language_instruction_tensor.clone(),
            ds.INITIAL_LANGUAGE_LENGTH: language_instruction_length.clone(),
        }
        collector.set_initial_state(initial_state)


    # 每个 episode 起点都记录 initial_state，与官方 replay_demos 读取逻辑对齐。
    capture_initial_state_for_episode()

    last_obs_dict = None
    last_action = torch.zeros((1, 8), dtype=torch.float32, device=device)
    last_reward = torch.zeros((1,), dtype=torch.float32, device=device)
    last_state_dict = None
    completed_demo_slots = 0

    # 使用仿真步长做控制解耦，避免 wall-clock sleep 导致 UI 阻塞。
    control_hz = max(1, int(task_preset.control_hz))
    vision_hz_arg = max(1, int(task_preset.vision_hz))
    vision_hz = min(vision_hz_arg, control_hz)
    if vision_hz_arg > control_hz:
        print(
            f"[WARN] vision_hz ({vision_hz_arg}) > control_hz ({control_hz}). "
            f"Clamp vision_hz to {vision_hz}."
        )
    sim_dt = float(sim.cfg.dt)
    control_decimation = max(1, int(round((1.0 / control_hz) / sim_dt)))
    vision_decimation = max(1, int(round(control_hz / vision_hz)))
    sim_step_count = 0
    control_step_count = 0
    episode_start_wall_time = time.perf_counter()
    last_rgb_top = None
    last_rgb_main = None
    last_rgb_wrist = None
    last_vision_control_step = -1
    vision_frame_counter = 0
    print(f"[INFO] Language instruction: {instruction_text}")
    print(
        f"[INFO] Loop decimation: sim_dt={sim_dt:.4f}s, "
        f"control_every={control_decimation} sim steps (~{control_hz}Hz), "
        f"vision_every={vision_decimation} control steps (~{vision_hz}Hz)."
    )
    # ==============================================================


    # ── 8. 主循环 ──────────────────────────────────────────────────────
    try:
        while simulation_app.is_running() and not should_exit:
            # 处理“保存并下一条”或“跳过并下一条”逻辑
            if should_reset or should_skip:
                finalize_mode = "save" if should_reset else "skip"

                if finalize_mode == "save":
                    # 对齐官方 record_demos：复位前导出一个成功 episode。
                    if last_obs_dict is not None and last_state_dict is not None:
                        done_flag = torch.ones((1,), device=device, dtype=torch.bool)
                        collector.add_step(last_obs_dict, last_action, last_reward, done_flag, last_state_dict)
                        if collector.export_episode(success=True):
                            print(
                                f"\n[INFO] Saved demo slot {completed_demo_slots + 1}"
                                f"/{collector.num_demos if collector.num_demos > 0 else '?'}"
                            )
                    else:
                        print("\n[INFO] No trajectory data yet. This slot will be counted and skipped.")
                else:
                    print(
                        f"\n[INFO] Skipped demo slot {completed_demo_slots + 1}"
                        f"/{collector.num_demos if collector.num_demos > 0 else '?'} (discard current trajectory)."
                    )
                    collector.reset_episode()
                    last_obs_dict = None
                    last_state_dict = None

                completed_demo_slots += 1

                # 达到设定的采集槽位数量后退出；跳过的槽位同样计入进度。
                if collector.num_demos > 0 and completed_demo_slots >= collector.num_demos:
                    print("[INFO] Data Collection Completed by demo slots. Exiting...")
                    break

                if virtual_button_lid_opener is not None:
                    virtual_button_lid_opener.reset_to_default_closed_state()

                sim.reset()
                robot.reset()
                if sensor_cameras:
                    for sensor in sensor_cameras.values():
                        sensor.reset()
                # R 重置仅复位关节，不改变根位置。
                # 不依赖 robot.data.default_joint_pos，始终写回任务预设缓存。
                robot.write_joint_state_to_sim(cached_init_joint_pos, cached_init_joint_vel)
                sim.step(render=True)
                robot.update(sim.cfg.dt)
                refresh_wrist_camera_viewport()

                target_pos = robot.data.body_pos_w[:, ee_body_id].clone()
                target_quat = robot.data.body_quat_w[:, ee_body_id].clone()
                teleop_interface.reset()
                should_reset = False

                collector.reset_episode()
                episode_start_wall_time = time.perf_counter()
                capture_initial_state_for_episode()
                last_obs_dict = None
                last_state_dict = None
                last_rgb_top = None
                last_rgb_main = None
                last_rgb_wrist = None
                last_vision_control_step = -1
                vision_frame_counter = 0
                should_reset = False
                should_skip = False

            if sim_step_count % control_decimation == 0:
                control_step_count += 1

                with torch.inference_mode():
                    # 获取手柄指令
                    gamepad_command, gripper_cmd = normalize_gamepad_output(teleop_interface.advance())

                    delta_pos = gamepad_command[:3].unsqueeze(0)
                    delta_rot = gamepad_command[3:6].unsqueeze(0)

                    # Se3GamepadCfg 已经应用了灵敏度，这里不再二次缩放，避免操作迟滞。
                    target_pos += delta_pos

                    # ── 位姿张量计算 (替代 Numpy/Scipy) ─────────────────────
                    if torch.norm(delta_rot) > 1e-6:
                        # 旋转向量转四元数
                        angle = torch.norm(delta_rot, dim=-1)
                        axis = delta_rot / angle
                        delta_quat = math_utils.quat_from_angle_axis(angle, axis)
                        target_quat = math_utils.quat_mul(target_quat, delta_quat)
                        target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)

                    # 下发给控制器
                    diff_ik_controller.set_command(torch.cat([target_pos, target_quat], dim=-1))

                    # ── 计算 IK 并获取关节指令 ───────────────────────────────
                    # 获取雅可比矩阵 (Shape: [1, num_bodies, 6, num_joints])
                    jacobian = robot.root_physx_view.get_jacobians()[:, ee_body_id, :, arm_joint_ids]

                    arm_joint_targets = diff_ik_controller.compute(
                        ee_pos=robot.data.body_pos_w[:, ee_body_id],
                        ee_quat=robot.data.body_quat_w[:, ee_body_id],
                        jacobian=jacobian,
                        joint_pos=robot.data.joint_pos[:, arm_joint_ids],
                    )

                    # ── 夹爪控制 (直接发送关节目标位置) ──────────────────────
                    gripper_targets = gripper_open_target if gripper_cmd else gripper_close_target
                    gripper_action = torch.tensor(
                        [[1.0 if gripper_cmd else 0.0]], dtype=torch.float32, device=device
                    )

                    # 记录动作为 8-DoF：7 维机械臂 + 1 维夹爪开合。
                    actual_action = torch.cat([arm_joint_targets, gripper_action], dim=-1)

                    # ── 下发控制指令 ─────────────────────────────────────────
                    if virtual_button_lid_opener is not None:
                        virtual_button_lid_opener.update(
                            get_virtual_button_probe_positions(),
                            dt_seconds=control_decimation * sim_dt,
                        )

                    robot.set_joint_position_target(arm_joint_targets, joint_ids=arm_joint_ids)
                    robot.set_joint_position_target(gripper_targets, joint_ids=gripper_joint_ids)
                    robot.write_data_to_sim()
                    # ==================== 新增：记录当前帧的数据 ====================
                    # 1. 组装观察数据 (Observation)
                    # 这里使用 state + vision 混合观测（仅采集主相机 + Top 相机）
                    should_capture_vision = ((control_step_count - 1) % vision_decimation == 0)
                    if should_capture_vision:
                        # 传入 sim_dt 用于刷新传感器状态
                        new_rgb_top = env_module.capture_rgb("top", sim_dt)
                        new_rgb_main = env_module.capture_rgb("main", sim_dt)
                        new_rgb_wrist = env_module.capture_rgb("wrist", sim_dt)
                        if new_rgb_top is not None and new_rgb_main is not None and new_rgb_wrist is not None:
                            last_rgb_top = new_rgb_top
                            last_rgb_main = new_rgb_main
                            last_rgb_wrist = new_rgb_wrist
                            last_vision_control_step = control_step_count
                            vision_frame_counter += 1

                    rgb_top = last_rgb_top
                    rgb_main = last_rgb_main
                    rgb_wrist = last_rgb_wrist

                    # 若视觉频率低于控制频率，复用最近一帧图像，并写入 freshness 标记。
                    has_vision = (rgb_top is not None) and (rgb_main is not None) and (rgb_wrist is not None)
                    vision_is_fresh = bool(last_vision_control_step == control_step_count)
                    vision_age_steps = (
                        control_step_count - last_vision_control_step if last_vision_control_step >= 0 else -1
                    )
                    can_record_step = has_vision

                    timestamp_sim_sec = torch.tensor(
                        [sim_step_count * sim_dt], dtype=torch.float32, device=device
                    )
                    timestamp_wall_sec = torch.tensor(
                        [time.perf_counter() - episode_start_wall_time],
                        dtype=torch.float32,
                        device=device,
                    )

                    # observation.state 对齐为 8 维: 7 维机械臂 + 1 维夹爪开合状态。
                    gripper_state = robot.data.joint_pos[:, gripper_joint_ids[0:1]].clone()
                    obs_joint_pos = torch.cat([robot.data.joint_pos[:, arm_joint_ids].clone(), gripper_state], dim=-1)

                    obs_dict = {
                        ds.OBS_ROBOT_JOINT_POS: obs_joint_pos,
                        ds.OBS_ROBOT_JOINT_VEL: robot.data.joint_vel[:, arm_joint_ids].clone(),
                        ds.OBS_ROBOT_EEF_POS: robot.data.body_pos_w[:, ee_body_id].clone(),
                        ds.OBS_ROBOT_EEF_QUAT: robot.data.body_quat_w[:, ee_body_id].clone(),
                        ds.OBS_TIMESTAMP_SIM_SEC: timestamp_sim_sec.clone(),
                        ds.OBS_TIMESTAMP_WALL_SEC: timestamp_wall_sec.clone(),
                        # 图像以 CPU uint8 存储，显著降低长轨迹录制时的 GPU 显存压力。
                        ds.OBS_RGB_TOP: rgb_top.detach().to(device="cpu", dtype=torch.uint8).clone() if rgb_top is not None else None,
                        ds.OBS_RGB_MAIN_45DEG: rgb_main.detach().to(device="cpu", dtype=torch.uint8).clone() if rgb_main is not None else None,
                        ds.OBS_RGB_WRIST: rgb_wrist.detach().to(device="cpu", dtype=torch.uint8).clone() if rgb_wrist is not None else None,
                        ds.OBS_VISION_IS_FRESH: torch.tensor([vision_is_fresh], dtype=torch.bool, device=device),
                        ds.OBS_VISION_AGE_STEPS: torch.tensor([vision_age_steps], dtype=torch.int32, device=device),
                        ds.OBS_VISION_FRAME_COUNTER: torch.tensor([vision_frame_counter], dtype=torch.int32, device=device),
                    }

                    # 2. 组装这一帧的伪奖励和完成状态
                    # 模仿学习通常不需要 reward，给 0 即可
                    reward = torch.zeros((1,), device=device, dtype=torch.float32)
                    done = torch.zeros((1,), device=device, dtype=torch.bool)

                    state_dict = {
                        ds.STATE_ROBOT_ROOT_STATE: robot.data.root_state_w[:, :13].clone(),
                        ds.STATE_ROBOT_JOINT_POS: robot.data.joint_pos.clone(),
                        ds.STATE_ROBOT_JOINT_VEL: robot.data.joint_vel.clone(),
                    }

                    if not can_record_step:
                        print("[WARN] RGB is unavailable yet. Skip dataset write on this control step.")
                    else:
                        # 3. 将 (obs, action, state) 录入 EpisodeData（对齐官方结构）
                        collector.add_step(obs_dict, actual_action, reward, done, state_dict)

                        last_obs_dict = {k: v.clone() for k, v in obs_dict.items()}
                        last_action = actual_action.clone()
                        last_reward = reward.clone()
                        last_state_dict = {k: v.clone() for k, v in state_dict.items()}
                    # ==============================================================

            # 物理步进与状态刷新
            sim.step(render=True)
            robot.update(sim.cfg.dt)
            sim_step_count += 1
    except KeyboardInterrupt:
        should_exit = True
        print("\n[INFO] KeyboardInterrupt received. Exiting gracefully...")

    if should_exit:
        print("[INFO] Exiting by user request.")

    # 关闭前取消未完成的异步任务，降低 close 时卡住概率。
    if dock_task is not None and not dock_task.done():
        dock_task.cancel()

    # 若用户直接退出，尽量保存当前未完结轨迹为一个失败 episode（不会影响已成功采集的数据）。
    if collector.has_data() and last_obs_dict is not None and last_state_dict is not None:
        try:
            done_flag = torch.ones((1,), device=device, dtype=torch.bool)
            collector.add_step(last_obs_dict, last_action, last_reward, done_flag, last_state_dict)
            if collector.export_episode(success=False):
                print("[INFO] Exported in-progress episode before shutdown (marked as failed).")
        except Exception as e:
            print(f"[WARN] Failed to export in-progress episode on exit: {e}")

    # ==================== 新增：关闭文件流 ====================
    # 确保 HDF5 文件尾部被正确封装，防止文件损坏
    collector.close()
    print("[INFO] Dataset saved and closed safely.")
    # ==========================================================
    print("[INFO] Closing simulation app...")
    simulation_app.close()


if __name__ == "__main__":
    main()
