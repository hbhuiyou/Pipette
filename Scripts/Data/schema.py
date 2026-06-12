from __future__ import annotations

import re

import h5py


DATA_GROUP = "data"
DEMO_PREFIX = "demo_"

ACTIONS = "actions"
OBS = "obs"
STATES = "states"
INITIAL_STATE = "initial_state"

OBS_RGB_TOP = "rgb_top"
OBS_RGB_MAIN = "rgb_main"
OBS_RGB_MAIN_45DEG = "rgb_main_45deg"
OBS_RGB_WRIST = "rgb_wrist"
OBS_ROBOT_JOINT_POS = "robot_joint_pos"
OBS_ROBOT_JOINT_VEL = "robot_joint_vel"
OBS_ROBOT_EEF_POS = "robot_eef_pos"
OBS_ROBOT_EEF_QUAT = "robot_eef_quat"
OBS_TIMESTAMP_SIM_SEC = "timestamp_sim_sec"
OBS_TIMESTAMP_WALL_SEC = "timestamp_wall_sec"
OBS_VISION_IS_FRESH = "vision_is_fresh"
OBS_VISION_AGE_STEPS = "vision_age_steps"
OBS_VISION_FRAME_COUNTER = "vision_frame_counter"

STATE_ROBOT_ROOT_STATE = "robot_root_state"
STATE_ROBOT_JOINT_POS = "robot_joint_pos"
STATE_ROBOT_JOINT_VEL = "robot_joint_vel"

INITIAL_ROBOT_ROOT_STATE = "robot_root_state"
INITIAL_ROBOT_JOINT_POS = "robot_joint_pos"
INITIAL_ROBOT_JOINT_VEL = "robot_joint_vel"
INITIAL_LANGUAGE_UTF8 = "language_instruction_utf8"
INITIAL_LANGUAGE_LENGTH = "language_instruction_length"

LEROBOT_IMAGE_TOP = "observation.images.top"
LEROBOT_IMAGE_MAIN = "observation.images.main"
LEROBOT_IMAGE_WRIST = "observation.images.wrist"
LEROBOT_STATE = "observation.state"
LEROBOT_ACTION = "action"
LEROBOT_LANGUAGE = "language_instruction"


def obs_path(name: str) -> str:
    return f"{OBS}/{name}"


def state_path(name: str) -> str:
    return f"{STATES}/{name}"


def initial_path(name: str) -> str:
    return f"{INITIAL_STATE}/{name}"


def demo_sort_key(name: str) -> int:
    match = re.fullmatch(rf"{DEMO_PREFIX}(\d+)", str(name))
    if match is None:
        return 10**12
    return int(match.group(1))


def demo_name(index: int) -> str:
    return f"{DEMO_PREFIX}{int(index)}"


def is_demo_name(name: str) -> bool:
    return re.fullmatch(rf"{DEMO_PREFIX}\d+", str(name)) is not None


def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    return sorted(
        [str(k) for k in data_group.keys() if is_demo_name(str(k))],
        key=demo_sort_key,
    )


def resolve_demo_root(h5f: h5py.File) -> h5py.Group:
    if DATA_GROUP in h5f and isinstance(h5f[DATA_GROUP], h5py.Group):
        return h5f[DATA_GROUP]
    return h5f


def require_data_group(h5f: h5py.File) -> h5py.Group:
    if DATA_GROUP not in h5f:
        raise RuntimeError(f"Input HDF5 missing top-level '{DATA_GROUP}' group.")
    data_group = h5f[DATA_GROUP]
    if not isinstance(data_group, h5py.Group):
        raise RuntimeError(f"Input HDF5 '{DATA_GROUP}' is not a group.")
    return data_group


def require_or_create_data_group(h5f: h5py.File) -> h5py.Group:
    if DATA_GROUP in h5f:
        data_group = h5f[DATA_GROUP]
        if not isinstance(data_group, h5py.Group):
            raise RuntimeError(f"Output HDF5 '{DATA_GROUP}' exists but is not a group.")
        return data_group
    return h5f.create_group(DATA_GROUP)


def next_demo_index(data_group: h5py.Group) -> int:
    demo_ids = [demo_sort_key(name) for name in data_group.keys()]
    demo_ids = [idx for idx in demo_ids if idx < 10**12]
    return (max(demo_ids) + 1) if demo_ids else 0


def get_main_rgb_dataset(obs_group: h5py.Group) -> h5py.Dataset:
    if OBS_RGB_MAIN_45DEG in obs_group:
        return obs_group[OBS_RGB_MAIN_45DEG]
    return obs_group[OBS_RGB_MAIN]


def required_obs_rgb_keys() -> tuple[str, str, str]:
    return (OBS_RGB_TOP, OBS_RGB_MAIN_45DEG, OBS_RGB_WRIST)


def hdf5_excluded_augmented_paths() -> set[str]:
    return {
        ACTIONS,
        obs_path(OBS_ROBOT_JOINT_POS),
        obs_path(OBS_ROBOT_JOINT_VEL),
        obs_path(OBS_ROBOT_EEF_POS),
        obs_path(OBS_ROBOT_EEF_QUAT),
        obs_path(OBS_TIMESTAMP_SIM_SEC),
        obs_path(OBS_TIMESTAMP_WALL_SEC),
        obs_path(OBS_RGB_TOP),
        obs_path(OBS_RGB_MAIN_45DEG),
        obs_path(OBS_RGB_MAIN),
        obs_path(OBS_RGB_WRIST),
        obs_path(OBS_VISION_IS_FRESH),
        obs_path(OBS_VISION_AGE_STEPS),
        obs_path(OBS_VISION_FRAME_COUNTER),
        state_path(STATE_ROBOT_ROOT_STATE),
        state_path(STATE_ROBOT_JOINT_POS),
        state_path(STATE_ROBOT_JOINT_VEL),
    }
