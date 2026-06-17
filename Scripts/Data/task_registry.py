from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskCameraSpec:
    name: str
    prim_path: str
    translation: tuple[float, float, float]
    rotation_xyz: tuple[float, float, float]
    focal_length: float
    enable_sensor_capture: bool = True


@dataclass(frozen=True)
class TubeEvalConfig:
    # Success when tube z exceeds (initial_z + success_height_delta)
    # and remains above for success_hold_seconds.
    success_height_delta: float = 0.02
    success_hold_seconds: float = 1.0
    timeout_seconds: float = 10.0
    # Fail if tube tilts beyond this angle from its initial orientation.
    tilt_fail_deg: float = 35.0
    success_y_rise_threshold: float | None = None
    success_z_drop_threshold: float | None = None
    # Optional fixed initial pose reference. If None, runtime reads from tube prim.
    initial_z: float | None = None
    initial_rotation_wxyz: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class CentrifugeEvalConfig:
    lid_prim_path: str = "/World/Centrifuge/models/Centrifuge_top"
    virtual_button_prim_path: str = ""
    lid_joint_prim_path: str = ""
    success_x_threshold_deg: float | None = None
    success_direction: str | None = None
    drive_target_success_threshold_deg: float = -39.0
    success_hold_seconds: float = 0.25
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class PipetteEvalConfig:
    success_xy_distance: float = 0.12
    low_z_threshold: float = 0.2
    timeout_seconds: float = 12.0


@dataclass(frozen=True)
class PlaceTubeOnBalanceEvalConfig:
    plate_prim_path: str = "/World/Electronic_Balance/root/ROOT/Plate"
    plate_radius_scale: float = 0.5
    tube_z_min: float = 0.63
    tube_z_max: float = 0.73
    success_hold_seconds: float = 1.0
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class PlacePipetteOnStandEvalConfig:
    success_y_threshold: float = 0.14
    success_z_threshold: float = 0.8
    success_hold_seconds: float = 1.0
    timeout_seconds: float = 12.0


@dataclass(frozen=True)
class WaterBathLidEvalConfig:
    lid_prim_path: str = "/World/Water_bath_1/root/lid"
    success_x_drop_threshold: float = 0.45
    success_z_rise_threshold: float = 0.1
    success_hold_seconds: float = 0.5
    timeout_seconds: float = 10.0


@dataclass(frozen=True)
class TaskPreset:
    task_id: str
    description: str
    usd_path: str
    env_name: str
    dataset_file: str
    language_instruction: str
    sensitivity: float = 4.0
    control_hz: int = 30
    vision_hz: int = 10
    camera_width: int = 400
    camera_height: int = 400
    main_cam_pitch_deg: float = 45.0
    camera_sensor_type: str = "camera"
    robot_prim_path: str = "/World/Franka"
    robot_init_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.024)
    robot_init_root_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    robot_init_joint_pos: dict[str, float] = field(default_factory=dict)
    camera_specs: tuple[TaskCameraSpec, ...] = field(default_factory=tuple)
    tube_prim_path: str = ""
    pipette_prim_path: str = ""
    petri_prim_path: str = ""
    petri_close_prim_path: str = ""
    petri_success_x_threshold: float = 0.5
    petri_success_z_threshold: float = 0.3
    petri_success_hold_seconds: float = 1.0
    petri_timeout_seconds: float = 10.0
    place_petri_success_x_min: float = 0.55
    place_petri_success_y_max: float = 0.1
    place_petri_success_z_threshold: float = 0.6
    place_petri_z_stable_seconds: float = 0.5
    place_petri_z_stable_tolerance: float = 0.002
    place_petri_timeout_seconds: float = 10.0
    tube_local_up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    tube_eval: TubeEvalConfig = field(default_factory=TubeEvalConfig)
    pipette_eval: PipetteEvalConfig = field(default_factory=PipetteEvalConfig)
    centrifuge_eval: CentrifugeEvalConfig = field(default_factory=CentrifugeEvalConfig)
    place_tube_on_balance_eval: PlaceTubeOnBalanceEvalConfig = field(default_factory=PlaceTubeOnBalanceEvalConfig)
    place_pipette_on_stand_eval: PlacePipetteOnStandEvalConfig = field(default_factory=PlacePipetteOnStandEvalConfig)
    water_bath_lid_eval: WaterBathLidEvalConfig = field(default_factory=WaterBathLidEvalConfig)


DEFAULT_TASK_ID = "pick_up_the_tube"
PICK_UP_PIPETTE_TASK_ID = "pick_up_the_pipette"
OPEN_CENTRIFUGE_LID_TASK_ID = "open_the_centrifuge_lid"
CLOSE_CENTRIFUGE_LID_TASK_ID = "close_the_centrifuge_lid"
TAKE_OUT_PETRI_DISH_TASK_ID = "take_out_the_petri_dish"
PLACE_PETRI_DISH_TASK_ID = "place_the_petri_dish"
PLACE_CENTRIFUGE_TUBE_ON_BALANCE_TASK_ID = "place_the_centrifuge_tube_on_the_balance"
PLACE_PIPETTE_ON_PIPETTE_STAND_TASK_ID = "place_the_pipette_on_the_pipette_stand"
TAKE_CENTRIFUGE_TUBE_FROM_BALANCE_TASK_ID = "take_the_centrifuge_tube_from_the_balance"
OPEN_WATER_BATH_LID_TASK_ID = "open_the_water_bath_lid"
CLOSE_SPECTROPHOTOMETER_TASK_ID = "close_the_spectrophotometer"


DEFAULT_FRANKA_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -0.785,
    "panda_joint3": 0.0,
    "panda_joint4": -2.356,
    "panda_joint5": 0.0,
    "panda_joint6": 1.571,
    "panda_joint7": 0.785,
    "panda_finger_joint.*": 0.04,
}

DEFAULT_MAIN_CAMERA = TaskCameraSpec( 
    name="main",
    prim_path="/World/MainOverviewCamera",
    translation=(0.4521, -1.1807, 2.4223),
    rotation_xyz=(35.0, 0.0, 0.0),
    focal_length=25.0,
    enable_sensor_capture=True,
)

DEFAULT_TOP_CAMERA = TaskCameraSpec(
    name="top",
    prim_path="/World/TopDownCamera",
    translation=(0.4291, 0.0035, 3.2409),
    rotation_xyz=(0.0, 0.0, 0.0),
    focal_length=45.0,
    enable_sensor_capture=True,
)

DEFAULT_WRIST_CAMERA = TaskCameraSpec(
    name="wrist",
    prim_path="/World/Franka/panda_hand/WristCamera",
    translation=(0.48854, -0.07732, -0.86398),
    rotation_xyz=(156.787, 3.822, 87.233),
    focal_length=60.0,
    enable_sensor_capture=True,
)

DEFAULT_CAMERA_SPECS = (
    DEFAULT_MAIN_CAMERA,
    DEFAULT_TOP_CAMERA,
    DEFAULT_WRIST_CAMERA,
)


def make_task(
    task_id: str,
    description: str,
    usd_path: str,
    dataset_file: str,
    language_instruction: str,
    *,
    tube_prim_path: str = "",
    pipette_prim_path: str = "",
    petri_prim_path: str = "",
    petri_close_prim_path: str = "",
    petri_success_x_threshold: float = 0.5,
    petri_success_z_threshold: float = 0.3,
    petri_success_hold_seconds: float = 1.0,
    petri_timeout_seconds: float = 10.0,
    place_petri_success_x_min: float = 0.55,
    place_petri_success_y_max: float = 0.1,
    place_petri_success_z_threshold: float = 0.6,
    place_petri_z_stable_seconds: float = 0.5,
    place_petri_z_stable_tolerance: float = 0.002,
    place_petri_timeout_seconds: float = 10.0,
    tube_eval: TubeEvalConfig | None = None,
    pipette_eval: PipetteEvalConfig | None = None,
    centrifuge_eval: CentrifugeEvalConfig | None = None,
    place_tube_on_balance_eval: PlaceTubeOnBalanceEvalConfig | None = None,
    place_pipette_on_stand_eval: PlacePipetteOnStandEvalConfig | None = None,
    water_bath_lid_eval: WaterBathLidEvalConfig | None = None,
) -> TaskPreset:
    return TaskPreset(
        task_id=task_id,
        description=description,
        usd_path=usd_path,
        env_name="MyCustomGraspTask",
        dataset_file=dataset_file,
        language_instruction=language_instruction,
        main_cam_pitch_deg=45.0,
        camera_sensor_type="camera",
        robot_init_root_pos=(0.0, 0.0, 0.024),
        robot_init_root_rot=(1.0, 0.0, 0.0, 0.0),
        robot_init_joint_pos=dict(DEFAULT_FRANKA_JOINT_POS),
        camera_specs=DEFAULT_CAMERA_SPECS,
        tube_prim_path=tube_prim_path,
        pipette_prim_path=pipette_prim_path,
        petri_prim_path=petri_prim_path,
        petri_close_prim_path=petri_close_prim_path,
        petri_success_x_threshold=float(petri_success_x_threshold),
        petri_success_z_threshold=float(petri_success_z_threshold),
        petri_success_hold_seconds=float(petri_success_hold_seconds),
        petri_timeout_seconds=float(petri_timeout_seconds),
        place_petri_success_x_min=float(place_petri_success_x_min),
        place_petri_success_y_max=float(place_petri_success_y_max),
        place_petri_success_z_threshold=float(place_petri_success_z_threshold),
        place_petri_z_stable_seconds=float(place_petri_z_stable_seconds),
        place_petri_z_stable_tolerance=float(place_petri_z_stable_tolerance),
        place_petri_timeout_seconds=float(place_petri_timeout_seconds),
        tube_local_up_axis=(0.0, 0.0, 1.0),
        tube_eval=tube_eval if tube_eval is not None else TubeEvalConfig(),
        pipette_eval=pipette_eval if pipette_eval is not None else PipetteEvalConfig(),
        centrifuge_eval=centrifuge_eval if centrifuge_eval is not None else CentrifugeEvalConfig(),
        place_tube_on_balance_eval=(
            place_tube_on_balance_eval
            if place_tube_on_balance_eval is not None
            else PlaceTubeOnBalanceEvalConfig()
        ),
        place_pipette_on_stand_eval=(
            place_pipette_on_stand_eval
            if place_pipette_on_stand_eval is not None
            else PlacePipetteOnStandEvalConfig()
        ),
        water_bath_lid_eval=(
            water_bath_lid_eval
            if water_bath_lid_eval is not None
            else WaterBathLidEvalConfig()
        ),
    )


TASK_PRESETS: dict[str, TaskPreset] = {
    DEFAULT_TASK_ID: make_task(
        task_id=DEFAULT_TASK_ID,
        description="Keyboard teleoperation data collection for tube pickup task.",
        usd_path="Asset/Scene/lab_0.usd",
        dataset_file="./datasets/pick_up_the_tube.hdf5",
        language_instruction="pick_up_the_tube",
        tube_prim_path="/World/Tube/Xform",
        tube_eval=TubeEvalConfig(
            success_height_delta=0.02,
            success_hold_seconds=0.5,
            timeout_seconds=7.0,
            tilt_fail_deg=35.0,
            initial_z=None,
            initial_rotation_wxyz=None,
        ),
    ),
    PICK_UP_PIPETTE_TASK_ID: make_task(
        task_id=PICK_UP_PIPETTE_TASK_ID,
        description="Keyboard teleoperation data collection for pipette pickup task.",
        usd_path="Asset/Scene/lab_1.usd",
        dataset_file="./datasets/pick_up_the_pipette.hdf5",
        language_instruction="pick_up_the_pipette",
        pipette_prim_path="/World/Pipette",
        petri_prim_path="/World/Petri",
        pipette_eval=PipetteEvalConfig(
            success_xy_distance=0.12,
            low_z_threshold=0.6,
            timeout_seconds=12.0,
        ),
    ),
    OPEN_CENTRIFUGE_LID_TASK_ID: make_task(
        task_id=OPEN_CENTRIFUGE_LID_TASK_ID,
        description="Keyboard teleoperation data collection for centrifuge lid opening task.",
        usd_path="Asset/Scene/lab_2.usd",
        dataset_file="./datasets/open_the_centrifuge_lid.hdf5",
        language_instruction="open_the_centrifuge_lid",
        centrifuge_eval=CentrifugeEvalConfig(
            lid_prim_path="/World/Centrifuge_1/models/Centrifuge_top",
            virtual_button_prim_path="/World/Centrifuge_1/VirtualButton_OpenLid",
            lid_joint_prim_path="/World/Centrifuge_1/models/Centrifuge_top/RevoluteJoint",
            drive_target_success_threshold_deg=-39.0,
            success_hold_seconds=0.25,
            timeout_seconds=10.0,
        ),
    ),
    CLOSE_CENTRIFUGE_LID_TASK_ID: make_task(
        task_id=CLOSE_CENTRIFUGE_LID_TASK_ID,
        description="Keyboard teleoperation data collection for centrifuge lid closing task.",
        usd_path="Asset/Scene/lab_3.usd",
        dataset_file="./datasets/close_the_centrifuge_lid.hdf5",
        language_instruction="close_the_centrifuge_lid",
        centrifuge_eval=CentrifugeEvalConfig(
            lid_prim_path="/World/Centrifuge/models/Centrifuge_top",
            success_x_threshold_deg=-46.0,
            success_direction="greater",
            success_hold_seconds=0.25,
            timeout_seconds=10.0,
        ),
    ),
    TAKE_OUT_PETRI_DISH_TASK_ID: make_task(
        task_id=TAKE_OUT_PETRI_DISH_TASK_ID,
        description="Keyboard teleoperation data collection for petri dish take-out task.",
        usd_path="Asset/Scene/lab_4.usd",
        dataset_file="./datasets/take_out_the_petri_dish.hdf5",
        language_instruction="take_out_the_petri_dish",
        petri_close_prim_path="/World/Petri_close",
        petri_success_x_threshold=0.6,
        petri_success_z_threshold=0.6,
        petri_success_hold_seconds=0.5,             #增强为0.5，评估为1
        petri_timeout_seconds=10.0,
    ),
    PLACE_PETRI_DISH_TASK_ID: make_task(
        task_id=PLACE_PETRI_DISH_TASK_ID,
        description="Keyboard teleoperation data collection for petri dish placement task.",
        usd_path="Asset/Scene/lab_5.usd",
        dataset_file="./datasets/place_the_petri_dish.hdf5",
        language_instruction="place_the_petri_dish",
        petri_close_prim_path="/World/Petri_close",
        place_petri_success_x_min=0.55,
        place_petri_success_y_max=0.1,
        place_petri_success_z_threshold=0.6,
        place_petri_z_stable_seconds=0.5,            #增强为0.5
        place_petri_z_stable_tolerance=0.002,
        place_petri_timeout_seconds=10.0,
    ),
    PLACE_CENTRIFUGE_TUBE_ON_BALANCE_TASK_ID: make_task(
        task_id=PLACE_CENTRIFUGE_TUBE_ON_BALANCE_TASK_ID,
        description="Keyboard teleoperation data collection for placing a centrifuge tube on an electronic balance.",
        usd_path="Asset/Scene/lab_6.usd",
        dataset_file="./datasets/place_the_centrifuge_tube_on_the_balance.hdf5",
        language_instruction="place_the_centrifuge_tube_on_the_balance",
        tube_prim_path="/World/Centrifuge_tube",
        place_tube_on_balance_eval=PlaceTubeOnBalanceEvalConfig(
            plate_prim_path="/World/Electronic_Balance/root/ROOT/Plate",
            plate_radius_scale=0.5,
            tube_z_min=0.63,
            tube_z_max=0.73,
            success_hold_seconds=1.0,               #增强0.7
            timeout_seconds=10.0,
        ),
    ),
    PLACE_PIPETTE_ON_PIPETTE_STAND_TASK_ID: make_task(
        task_id=PLACE_PIPETTE_ON_PIPETTE_STAND_TASK_ID,
        description="Keyboard teleoperation data collection for placing a pipette on a pipette stand.",
        usd_path="Asset/Scene/lab_7.usd",
        dataset_file="./datasets/place_the_pipette_on_the_pipette_stand.hdf5",
        language_instruction="place_the_pipette_on_the_pipette_stand",
        pipette_prim_path="/World/Pipette",
        petri_prim_path="/World/Pipette_Stand",
        pipette_eval=PipetteEvalConfig(
            success_xy_distance=0.12,
            low_z_threshold=0.6,
            timeout_seconds=12.0,
        ),
        place_pipette_on_stand_eval=PlacePipetteOnStandEvalConfig(
            success_y_threshold=0.14,
            success_z_threshold=0.8,
            success_hold_seconds=1.0,
            timeout_seconds=12.0,
        ),
    ),
    TAKE_CENTRIFUGE_TUBE_FROM_BALANCE_TASK_ID: make_task(
        task_id=TAKE_CENTRIFUGE_TUBE_FROM_BALANCE_TASK_ID,
        description="Keyboard teleoperation data collection for taking a centrifuge tube away from an electronic balance.",
        usd_path="Asset/Scene/lab_8.usd",
        dataset_file="./datasets/take_the_centrifuge_tube_from_the_balance.hdf5",
        language_instruction="take_the_centrifuge_tube_from_the_balance",
        tube_prim_path="/World/Centrifuge_tube",
        tube_eval=TubeEvalConfig(
            success_height_delta=0.02,
            success_hold_seconds=1.2,
            timeout_seconds=10.0,
            tilt_fail_deg=10.0,
            success_y_rise_threshold=0.25,
            success_z_drop_threshold=0.12,
            initial_z=None,
            initial_rotation_wxyz=None,
        ),
    ),
    OPEN_WATER_BATH_LID_TASK_ID: make_task(
        task_id=OPEN_WATER_BATH_LID_TASK_ID,
        description="Keyboard teleoperation data collection for opening the water bath lid.",
        usd_path="Asset/Scene/lab_9.usd",
        dataset_file="./datasets/open_the_water_bath_lid.hdf5",
        language_instruction="open_the_water_bath_lid",
        water_bath_lid_eval=WaterBathLidEvalConfig(
            lid_prim_path="/World/Water_bath_1/root/lid",
            success_x_drop_threshold=0.30,
            success_z_rise_threshold=0.1,
            success_hold_seconds=0.3,
            timeout_seconds=10.0,
        ),
    ),
    CLOSE_SPECTROPHOTOMETER_TASK_ID: make_task(
        task_id=CLOSE_SPECTROPHOTOMETER_TASK_ID,
        description="Keyboard teleoperation data collection for closing the spectrophotometer.",
        usd_path="Asset/Scene/lab_10.usd",
        dataset_file="./datasets/close_the_spectrophotometer.hdf5",
        language_instruction="close_the_spectrophotometer",
        centrifuge_eval=CentrifugeEvalConfig(
            lid_prim_path="/World/Ultramicro_spectrophotometer/root/lid",
            success_x_threshold_deg=90.0,
            success_direction="greater",
            success_hold_seconds=0.5,
            timeout_seconds=10.0,
        ),
    ),
}


def get_task_preset(task_id: str) -> TaskPreset:
    if task_id not in TASK_PRESETS:
        known = ", ".join(sorted(TASK_PRESETS.keys()))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {known}")
    return TASK_PRESETS[task_id]


def list_task_presets() -> list[TaskPreset]:
    return [TASK_PRESETS[k] for k in sorted(TASK_PRESETS.keys())]


def get_tube_tracking_spec(task_id: str) -> tuple[str, tuple[float, float, float]]:
    preset = get_task_preset(task_id)
    return preset.tube_prim_path, preset.tube_local_up_axis


def get_tube_eval_config(task_id: str) -> TubeEvalConfig:
    preset = get_task_preset(task_id)
    return preset.tube_eval


def get_task_timeout_seconds(task_preset: TaskPreset) -> float:
    task_id = str(task_preset.task_id).lower()
    if task_id == PLACE_CENTRIFUGE_TUBE_ON_BALANCE_TASK_ID:
        return float(task_preset.place_tube_on_balance_eval.timeout_seconds)
    if task_id == PLACE_PIPETTE_ON_PIPETTE_STAND_TASK_ID:
        return float(task_preset.place_pipette_on_stand_eval.timeout_seconds)
    if task_id == CLOSE_SPECTROPHOTOMETER_TASK_ID:
        return float(task_preset.centrifuge_eval.timeout_seconds)
    if task_id == OPEN_WATER_BATH_LID_TASK_ID:
        return float(task_preset.water_bath_lid_eval.timeout_seconds)
    if "pipette" in task_id:
        return float(task_preset.pipette_eval.timeout_seconds)
    if task_id == TAKE_OUT_PETRI_DISH_TASK_ID:
        return float(task_preset.petri_timeout_seconds)
    if task_id == PLACE_PETRI_DISH_TASK_ID:
        return float(task_preset.place_petri_timeout_seconds)
    if task_id in (OPEN_CENTRIFUGE_LID_TASK_ID, CLOSE_CENTRIFUGE_LID_TASK_ID):
        return float(task_preset.centrifuge_eval.timeout_seconds)
    return float(task_preset.tube_eval.timeout_seconds)
    
