from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PIPETTE_ROOT = Path(__file__).resolve().parents[2]


def resolve_task_usd_path(path_text: str) -> Path:
	path = Path(path_text).expanduser()
	resolved = path if path.is_absolute() else (PIPETTE_ROOT / path).resolve()
	if not resolved.is_file():
		raise FileNotFoundError(f"Task USD scene not found: {resolved}")
	return resolved


@dataclass
class CameraPrimSpec:
	name: str
	prim_path: str
	translation: tuple[float, float, float]
	rotation_xyz: tuple[float, float, float]
	focal_length: float
	enable_sensor_capture: bool = True


@dataclass
class EnvironmentModuleConfig:
	usd_path: str
	camera_width: int = 400
	camera_height: int = 400
	camera_sensor_type: str = "camera"
	warmup_render_steps: int = 6
	reset_robot_root_pose: bool = False
	camera_specs: list[CameraPrimSpec] = field(default_factory=list)


@dataclass
class FrankaSimulationSession:
	env_module: "IsaacLabEnvironmentModule"
	sim: Any
	robot: Any
	arm_joint_ids: Any
	gripper_joint_ids: Any
	ee_body_id: int | None
	sensor_cameras: dict[str, Any]
	camera_specs: list[CameraPrimSpec]


def log_robot_root_alignment(robot: Any, prefix: str) -> None:
	"""Print root pose alignment between current and task default state for reset diagnostics."""
	try:
		import torch

		root_pos = getattr(robot.data, "root_pos_w", None)
		default_root = getattr(robot.data, "default_root_state", None)
		if root_pos is None or default_root is None:
			return
		target_pos = default_root[:, :3]
		pos_err = torch.norm(root_pos - target_pos, dim=1)
		max_err = float(pos_err.max().item())
		print(f"[INFO] {prefix} root_pos={root_pos[0].tolist()} target={target_pos[0].tolist()} err={max_err:.4f}m")
	except Exception:
		return


def reset_episode_state(env_module: Any, warmup_steps: int, enforce_root_pose_reset: bool) -> None:
	"""Reset simulation, robot joints, sensors, and run warmup steps for an evaluation episode."""
	env_module.sim.reset()
	env_module.robot.reset()

	if enforce_root_pose_reset:
		env_module.robot.write_root_pose_to_sim(env_module.robot.data.default_root_state[:, :7])

	env_module.robot.write_joint_state_to_sim(
		env_module.robot.data.default_joint_pos,
		env_module.robot.data.default_joint_vel,
	)
	env_module.sim.step(render=True)
	env_module.robot.update(env_module.sim.cfg.dt)

	for sensor in env_module.sensor_cameras.values():
		sensor.reset()

	for _ in range(max(0, int(warmup_steps))):
		env_module.sim.step(render=True)
		env_module.robot.update(env_module.sim.cfg.dt)


def apply_camera_launch_workarounds(args_cli: Any) -> Any:
	"""Apply IsaacLab 2.3 and Isaac Sim 5.1 camera launch flags.

	Official docs and release notes recommend:
	1) enable camera rendering explicitly when using camera sensors
	2) use offscreen rendering for headless camera capture
	3) apply known render-loop workarounds for 5.1 regressions
	"""
	args_cli.enable_cameras = True

	if getattr(args_cli, "headless", False) and hasattr(args_cli, "offscreen_render"):
		args_cli.offscreen_render = True

	if hasattr(args_cli, "kit_args"):
		stable_kit_args = (
			" --/rtx/post/dlss/execMode=0"
			" --/app/runLoops/main/rateLimitEnabled=false"
			" --/app/runLoops/main/manualModeEnabled=true"
			" --enable omni.kit.loop-isaac"
		)
		args_cli.kit_args = f"{args_cli.kit_args or ''}{stable_kit_args}"

	return args_cli


def task_camera_specs(
	task_preset: Any,
	*,
	enable_sensor_capture: bool | None = None,
) -> list[CameraPrimSpec]:
	specs: list[CameraPrimSpec] = []
	for spec in task_preset.camera_specs:
		specs.append(
			CameraPrimSpec(
				name=spec.name,
				prim_path=spec.prim_path,
				translation=spec.translation,
				rotation_xyz=spec.rotation_xyz,
				focal_length=spec.focal_length,
				enable_sensor_capture=(
					bool(enable_sensor_capture)
					if enable_sensor_capture is not None
					else bool(spec.enable_sensor_capture)
				),
			)
		)
	return specs


def create_franka_simulation_session(
	task_preset: Any,
	*,
	camera_width: int | None = None,
	camera_height: int | None = None,
	camera_sensor_type: str | None = None,
	warmup_render_steps: int = 6,
	sim_dt: float = 1.0 / 60.0,
	render_interval: int = 4,
	use_fabric: bool = True,
	enable_sensor_capture: bool | None = None,
	create_camera_prims: bool = True,
	create_sensor_cameras: bool = True,
	reset_robot_root_pose: bool = False,
	find_ee_body: bool = True,
	configure_robot_cfg: Any | None = None,
) -> FrankaSimulationSession:
	"""Create the common IsaacLab + Franka session used by scripts.

	Call this only after AppLauncher has started the Isaac/Kit application.
	"""
	try:
		from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
	except Exception:
		from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

	active_camera_specs = task_camera_specs(
		task_preset,
		enable_sensor_capture=enable_sensor_capture,
	)
	task_usd_path = resolve_task_usd_path(task_preset.usd_path)
	env_cfg = EnvironmentModuleConfig(
		usd_path=str(task_usd_path),
		camera_width=max(32, int(camera_width if camera_width is not None else task_preset.camera_width)),
		camera_height=max(32, int(camera_height if camera_height is not None else task_preset.camera_height)),
		camera_sensor_type=str(camera_sensor_type if camera_sensor_type is not None else task_preset.camera_sensor_type),
		warmup_render_steps=int(warmup_render_steps),
		reset_robot_root_pose=bool(reset_robot_root_pose),
		camera_specs=active_camera_specs,
	)

	env_module = IsaacLabEnvironmentModule(env_cfg)
	sim = env_module.create_simulation(dt=float(sim_dt), render_interval=int(render_interval), use_fabric=bool(use_fabric))

	if hasattr(FRANKA_PANDA_HIGH_PD_CFG, "replace"):
		robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=task_preset.robot_prim_path)
	else:
		robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
		robot_cfg.prim_path = task_preset.robot_prim_path

	robot_cfg.init_state.pos = task_preset.robot_init_root_pos
	robot_cfg.init_state.rot = task_preset.robot_init_root_rot
	robot_cfg.init_state.joint_pos = dict(task_preset.robot_init_joint_pos)
	if configure_robot_cfg is not None:
		configure_robot_cfg(robot_cfg)

	# Increase gripper stiffness and effort to prevent objects from slipping
	gripper_actuator = robot_cfg.actuators.get("panda_gripper")
	if gripper_actuator is not None:
		if hasattr(gripper_actuator, "stiffness"):
			gripper_actuator.stiffness = 1e4
		if hasattr(gripper_actuator, "damping"):
			gripper_actuator.damping = 1e3
		if hasattr(gripper_actuator, "effort_limit"):
			gripper_actuator.effort_limit = 1000.0

	robot = env_module.create_robot(robot_cfg)
	env_module.initialize_robot_home_pose()

	if create_camera_prims:
		env_module.define_camera_prims()
	if create_sensor_cameras:
		env_module.create_sensor_cameras()

	arm_joint_ids = robot.find_joints("panda_joint.*")[0]
	gripper_joint_ids = robot.find_joints("panda_finger.*")[0]
	ee_body_id = robot.find_bodies("panda_hand")[0][0] if find_ee_body else None

	return FrankaSimulationSession(
		env_module=env_module,
		sim=sim,
		robot=robot,
		arm_joint_ids=arm_joint_ids,
		gripper_joint_ids=gripper_joint_ids,
		ee_body_id=ee_body_id,
		sensor_cameras=env_module.sensor_cameras,
		camera_specs=active_camera_specs,
	)


class IsaacLabEnvironmentModule:
	"""Reusable environment bootstrapper for IsaacLab 2.3 scripts.

	This module encapsulates:
	- opening a USD stage
	- creating SimulationContext
	- taking over robot articulation
	- creating camera prims and sensor cameras
	- warm-up and reset routines
	"""

	def __init__(self, cfg: EnvironmentModuleConfig):
		self.cfg = cfg
		self.sim = None
		self.robot = None
		self.sensor_cameras: dict[str, Any] = {}
		self.device: str | None = None
		self._camera_paths: dict[str, str] = {}

	def _robot_is_fixed_base(self) -> bool:
		if self.robot is None:
			return False
		for attr_name in ("is_fixed_base", "fixed_base"):
			value = getattr(self.robot, attr_name, None)
			if isinstance(value, bool):
				return value
			if hasattr(value, "item"):
				try:
					return bool(value.item())
				except Exception:
					pass
		return False

	def _should_reset_root_pose(self) -> bool:
		return bool(self.cfg.reset_robot_root_pose) and (not self._robot_is_fixed_base())

	def create_simulation(self, dt: float = 1.0 / 60.0, render_interval: int = 4, use_fabric: bool = True):
		import omni.usd
		import isaaclab.sim as sim_utils

		omni.usd.get_context().new_stage()
		stage = omni.usd.get_context().get_stage()
		stage.GetRootLayer().subLayerPaths.insert(0, self.cfg.usd_path)

		try:
			physx_cfg = sim_utils.PhysxCfg(enable_stabilization=True)
		except Exception:
			physx_cfg = None

		if physx_cfg is None:
			sim_cfg = sim_utils.SimulationCfg(
				dt=dt,
				render_interval=render_interval,
				use_fabric=use_fabric,
				render=sim_utils.RenderCfg(enable_translucency=True),
			)
		else:
			sim_cfg = sim_utils.SimulationCfg(
				dt=dt,
				render_interval=render_interval,
				use_fabric=use_fabric,
				physx=physx_cfg,
				render=sim_utils.RenderCfg(enable_translucency=True),
			)
		self.sim = sim_utils.SimulationContext(sim_cfg)
		self.device = self.sim.device
		return self.sim

	def create_robot(self, robot_cfg: Any):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before robot initialization.")

		from isaaclab.assets import Articulation

		self.robot = Articulation(cfg=robot_cfg)
		return self.robot

	def initialize_robot_home_pose(self):
		if self.sim is None or self.robot is None:
			raise RuntimeError("Simulation and robot must be created before home pose init.")

		self.sim.reset()
		self.robot.update(self.sim.cfg.dt)

		if self._should_reset_root_pose():
			self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[:, :7])
		self.robot.write_joint_state_to_sim(
			self.robot.data.default_joint_pos,
			self.robot.data.default_joint_vel,
		)

		self.sim.step(render=True)
		self.robot.update(self.sim.cfg.dt)

	@staticmethod
	def _find_xform_op(xformable, op_type):
		for op in xformable.GetOrderedXformOps():
			if op.GetOpType() == op_type:
				return op
		return None

	@staticmethod
	def _rotation_xyz_to_quat(rotation_xyz):
		from pxr import Gf

		rot_x = Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), rotation_xyz[0])
		rot_y = Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), rotation_xyz[1])
		rot_z = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rotation_xyz[2])
		return (rot_x * rot_y * rot_z).GetQuat()

	def _set_camera_transform(self, cam, translation, rotation_xyz):
		from pxr import Gf, UsdGeom

		xformable = UsdGeom.Xformable(cam.GetPrim())

		translate_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeTranslate)
		if translate_op is None:
			translate_op = xformable.AddTranslateOp()
		translate_op.Set(Gf.Vec3d(*translation))

		orient_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeOrient)
		if orient_op is not None:
			orient_op.Set(self._rotation_xyz_to_quat(rotation_xyz))
			return

		rotate_xyz_op = self._find_xform_op(xformable, UsdGeom.XformOp.TypeRotateXYZ)
		if rotate_xyz_op is None:
			rotate_xyz_op = xformable.AddRotateXYZOp()
		rotate_xyz_op.Set(Gf.Vec3d(*rotation_xyz))

	def refresh_camera_prim(
		self,
		camera_name: str,
		translation_offset: tuple[float, float, float] | None = None,
	):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before camera prim refresh.")

		import omni.usd
		import isaaclab.sim as sim_utils
		from pxr import UsdGeom

		stage = omni.usd.get_context().get_stage()

		for spec in self.cfg.camera_specs:
			if spec.name == camera_name:
				translation = spec.translation
				if translation_offset is not None:
					if len(translation_offset) != 3:
						raise ValueError("Camera translation_offset must contain exactly 3 values.")
					translation = tuple(
						float(spec.translation[index]) + float(translation_offset[index])
						for index in range(3)
					)

				cam = UsdGeom.Camera.Define(stage, spec.prim_path)
				self._set_camera_transform(cam, translation, spec.rotation_xyz)
				cam.GetFocalLengthAttr().Set(spec.focal_length)

				cam_prim = cam.GetPrim()
				if not sim_utils.standardize_xform_ops(cam_prim):
					raise RuntimeError(
						f"Failed to standardize camera xform ops at '{spec.prim_path}'."
					)
				self._camera_paths[spec.name] = spec.prim_path
				return spec.prim_path

		raise KeyError(f"Unknown camera spec name '{camera_name}'.")

	def define_camera_prims(self):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before camera prim definition.")

		for spec in self.cfg.camera_specs:
			self.refresh_camera_prim(spec.name)

	def create_sensor_cameras(self):
		if self.sim is None:
			raise RuntimeError("Simulation must be created before sensor camera setup.")

		import omni.usd
		import isaaclab.sim as sim_utils

		try:
			from isaaclab.sensors import Camera as IsaacSensorCamera
			from isaaclab.sensors import CameraCfg
			from isaaclab.sensors import TiledCamera
			from isaaclab.sensors import TiledCameraCfg
		except Exception as exc:
			raise RuntimeError(
				"isaaclab.sensors.Camera/TiledCamera is unavailable in current environment."
			) from exc

		sensor_type = self.cfg.camera_sensor_type.lower()
		if sensor_type not in {"camera", "tiled"}:
			raise ValueError("camera_sensor_type must be 'camera' or 'tiled'.")

		if sensor_type == "tiled":
			cfg_cls = TiledCameraCfg
			sensor_cls = TiledCamera
		else:
			cfg_cls = CameraCfg
			sensor_cls = IsaacSensorCamera

		for _ in range(3):
			self.sim.render()

		self.sensor_cameras = {}
		stage = omni.usd.get_context().get_stage()
		for spec in self.cfg.camera_specs:
			if not spec.enable_sensor_capture:
				continue

			cam_prim = stage.GetPrimAtPath(spec.prim_path)
			if not cam_prim.IsValid():
				raise RuntimeError(f"Camera prim '{spec.prim_path}' does not exist on stage.")
			if not sim_utils.standardize_xform_ops(cam_prim):
				raise RuntimeError(
					f"Failed to standardize camera xform ops at '{spec.prim_path}' before sensor initialization."
				)

			cam_cfg = cfg_cls(
				prim_path=spec.prim_path,
				update_period=0.0,
				height=self.cfg.camera_height,
				width=self.cfg.camera_width,
				data_types=["rgb"],
				spawn=None,
			)

			sensor = sensor_cls(cfg=cam_cfg)
			self.sensor_cameras[spec.name] = sensor

		self.sim.reset()
		if self.robot is not None:
			self.robot.update(self.sim.cfg.dt)
			if self._should_reset_root_pose():
				self.robot.write_root_pose_to_sim(self.robot.data.default_root_state[:, :7])
			self.robot.write_joint_state_to_sim(
				self.robot.data.default_joint_pos,
				self.robot.data.default_joint_vel,
			)
		self.sim.step(render=True)
		if self.robot is not None:
			self.robot.update(self.sim.cfg.dt)

		for sensor in self.sensor_cameras.values():
			sensor.reset()

		for _ in range(max(0, int(self.cfg.warmup_render_steps))):
			self.sim.render()

		return self.sensor_cameras

	def capture_rgb(self, camera_name: str, dt: float | None = None):
		sensor = self.sensor_cameras.get(camera_name)
		if sensor is None:
			return None

		if dt is None:
			dt = float(self.sim.cfg.dt)

		sensor.update(dt)
		rgb = sensor.data.output.get("rgb")
		if rgb is None or rgb.numel() == 0:
			return None

		return rgb[0].clone()

	@property
	def camera_paths(self) -> dict[str, str]:
		return dict(self._camera_paths)

