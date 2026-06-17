"""
IsaacLab HDF5 to LeRobot Dataset Converter.

Default behavior keeps only frames marked by obs/vision_is_fresh and writes
the dataset at 10 FPS. For experiments trained directly from 30 Hz augmented
data, run with --frame-filter all --fps 30 and use --policy-hz 30 at inference.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import schema as ds


def _load_conversion_dependencies():
    import h5py
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tqdm import tqdm

    return h5py, np, torch, LeRobotDataset, tqdm


def _decode_instruction(lang_bytes: np.ndarray) -> str:
    """Decode the HDF5 uint8 UTF-8 instruction tensor."""
    if lang_bytes.size == 0:
        return ""
    arr = lang_bytes.flatten()
    return bytes(arr.tolist()).decode("utf-8")


def _squeeze_leading_env_dim(x: np.ndarray) -> np.ndarray:
    """Support both (T, D) and IsaacLab-style (T, 1, D) tensors."""
    if x.ndim >= 3 and x.shape[1] == 1:
        return np.squeeze(x, axis=1)
    return x


def _should_keep_frame(
    is_fresh_all: np.ndarray | None,
    index: int,
    frame_filter: str,
    stride: int,
) -> bool:
    if frame_filter == "all":
        return True
    if frame_filter == "stride":
        return (int(index) % max(1, int(stride))) == 0
    if is_fresh_all is None:
        return True
    return bool(is_fresh_all[index].reshape(-1)[0])


def _load_main_rgb(ep_group: h5py.Group) -> np.ndarray:
    return ds.get_main_rgb_dataset(ep_group[ds.OBS])[:]


def _infer_source_fps(ep_group: h5py.Group) -> float | None:
    if ds.OBS_TIMESTAMP_SIM_SEC not in ep_group[ds.OBS]:
        return None
    timestamps = ep_group[ds.OBS][ds.OBS_TIMESTAMP_SIM_SEC][:]
    values = timestamps.reshape(timestamps.shape[0], -1)[:, 0].astype(np.float64, copy=False)
    if values.size < 3:
        return None
    deltas = np.diff(values)
    deltas = deltas[np.isfinite(deltas) & (deltas > 1.0e-6)]
    if deltas.size == 0:
        return None
    return float(1.0 / np.median(deltas))


def _resolve_episode_filter(
    ep_name: str,
    ep_group: h5py.Group,
    is_fresh_all: np.ndarray | None,
    frame_filter: str,
    stride: int,
    fps: int,
) -> tuple[str, int]:
    if frame_filter != "fresh" or is_fresh_all is None:
        return frame_filter, stride

    fresh_flat = is_fresh_all.reshape(-1).astype(bool)
    if fresh_flat.size == 0 or not bool(np.all(fresh_flat)):
        return frame_filter, stride

    source_fps = _infer_source_fps(ep_group)
    if source_fps is None:
        source_fps = 30.0

    if source_fps <= float(fps) * 1.5:
        return frame_filter, stride

    auto_stride = max(1, int(stride))
    if auto_stride <= 1:
        auto_stride = max(1, int(round(source_fps / max(1, int(fps)))))
    print(
        f"[WARN] {ep_name}: vision_is_fresh is all True but source_fps~{source_fps:.2f} > target fps={fps}. "
        f"Auto downsample with stride={auto_stride}."
    )
    return "stride", auto_stride


def convert_isaac_to_lerobot(
    hdf5_path: str,
    repo_id: str,
    local_dir: str,
    fps: int = 10,
    frame_filter: str = "fresh",
    stride: int = 3,
):
    h5py, np, torch, LeRobotDataset, tqdm = _load_conversion_dependencies()

    print(f"[INFO] Opening HDF5 file: {hdf5_path}")
    print(f"[INFO] LeRobot fps: {int(fps)}")
    print(f"[INFO] Frame filter: {frame_filter}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        root=local_dir,
        features={
            ds.LEROBOT_IMAGE_TOP: {
                "dtype": "video",
                "shape": (3, 400, 400),
                "names": ["channels", "height", "width"],
            },
            ds.LEROBOT_IMAGE_MAIN: {
                "dtype": "video",
                "shape": (3, 400, 400),
                "names": ["channels", "height", "width"],
            },
            ds.LEROBOT_IMAGE_WRIST: {
                "dtype": "video",
                "shape": (3, 400, 400),
                "names": ["channels", "height", "width"],
            },
            ds.LEROBOT_STATE: {
                "dtype": "float32",
                "shape": (8,),
                "names": ["arm_joints", "gripper"],
            },
            ds.LEROBOT_ACTION: {
                "dtype": "float32",
                "shape": (8,),
                "names": ["arm_joints", "gripper"],
            },
            ds.LEROBOT_LANGUAGE: {"dtype": "string", "shape": (1,), "names": None},
        },
    )

    with h5py.File(hdf5_path, "r") as f:
        if ds.DATA_GROUP not in f:
            raise RuntimeError("Input HDF5 missing top-level 'data' group.")

        episodes = ds.sorted_demo_names(f[ds.DATA_GROUP])
        total_episodes = len(episodes)
        print(f"[INFO] Total episodes: {total_episodes}", flush=True)

        total_kept = 0
        for episode_index, ep_name in enumerate(tqdm(episodes, desc="Converting Episodes to MP4 & Parquet"), start=1):
            ep_group = f[ds.DATA_GROUP][ep_name]

            rgb_top_all = ep_group[ds.OBS][ds.OBS_RGB_TOP][:]
            rgb_main_all = _load_main_rgb(ep_group)
            rgb_wrist_all = ep_group[ds.OBS][ds.OBS_RGB_WRIST][:]
            joint_pos_all = _squeeze_leading_env_dim(ep_group[ds.OBS][ds.OBS_ROBOT_JOINT_POS][:])
            actions_all = _squeeze_leading_env_dim(ep_group[ds.ACTIONS][:])

            is_fresh_all = None
            if ds.OBS_VISION_IS_FRESH in ep_group[ds.OBS]:
                is_fresh_all = ep_group[ds.OBS][ds.OBS_VISION_IS_FRESH][:]

            episode_filter, episode_stride = _resolve_episode_filter(
                ep_name=ep_name,
                ep_group=ep_group,
                is_fresh_all=is_fresh_all,
                frame_filter=frame_filter,
                stride=stride,
                fps=int(fps),
            )

            lang_bytes = ep_group[ds.INITIAL_STATE][ds.INITIAL_LANGUAGE_UTF8][:]
            instruction = _decode_instruction(lang_bytes) or "pick_up_the_tube"

            kept_frames = 0
            for i in range(actions_all.shape[0]):
                if not _should_keep_frame(
                    is_fresh_all,
                    i,
                    frame_filter=episode_filter,
                    stride=episode_stride,
                ):
                    continue

                frame_dict = {
                    ds.LEROBOT_IMAGE_TOP: torch.from_numpy(np.transpose(rgb_top_all[i], (2, 0, 1))).clone(),
                    ds.LEROBOT_IMAGE_MAIN: torch.from_numpy(np.transpose(rgb_main_all[i], (2, 0, 1))).clone(),
                    ds.LEROBOT_IMAGE_WRIST: torch.from_numpy(np.transpose(rgb_wrist_all[i], (2, 0, 1))).clone(),
                    ds.LEROBOT_STATE: torch.from_numpy(joint_pos_all[i]).clone(),
                    ds.LEROBOT_ACTION: torch.from_numpy(actions_all[i]).clone(),
                    ds.LEROBOT_LANGUAGE: instruction,
                    "task": instruction,
                }
                dataset.add_frame(frame_dict)
                kept_frames += 1

            if kept_frames == 0:
                print(f"[WARN] {ep_name} kept 0 frames; skip save_episode().")
                print(
                    f"[PROGRESS] Converted episodes: {episode_index}/{total_episodes}, kept_frames={total_kept}",
                    flush=True,
                )
                continue

            total_kept += kept_frames
            dataset.save_episode()
            print(
                f"[PROGRESS] Converted episodes: {episode_index}/{total_episodes}, kept_frames={total_kept}",
                flush=True,
            )

    dataset.finalize()
    print(f"\n[SUCCESS] LeRobot Dataset created successfully at: {local_dir}")
    print(f"[INFO] Total kept frames: {total_kept}")


def build_args():
    parser = argparse.ArgumentParser(description="Convert IsaacLab HDF5 demonstrations to LeRobot format.")
    parser.add_argument("--hdf5-path", "--hdf5_path", type=str, required=True, help="Input HDF5 dataset path.")
    parser.add_argument("--repo-id", "--repo_id", type=str, default="local/isaaclab_pick_up_the_tube")
    parser.add_argument("--output-dir", "--output_dir", type=str, required=True, help="Output LeRobot dataset directory.")
    parser.add_argument("--fps", type=int, default=10, help="FPS written into the LeRobot dataset metadata.")
    parser.add_argument(
        "--frame-filter",
        "--frame_filter",
        type=str,
        default="fresh",
        choices=["fresh", "all", "stride"],
        help="fresh=keep obs/vision_is_fresh frames; all=keep every frame; stride=keep every Nth frame.",
    )
    parser.add_argument("--stride", type=int, default=3, help="Frame stride used when --frame-filter=stride.")
    return parser.parse_args()


def main() -> int:
    args = build_args()
    if not os.path.exists(args.hdf5_path):
        print(f"[ERROR] HDF5 file not found: {args.hdf5_path}")
        return 1

    convert_isaac_to_lerobot(
        hdf5_path=args.hdf5_path,
        repo_id=args.repo_id,
        local_dir=args.output_dir,
        fps=int(args.fps),
        frame_filter=str(args.frame_filter),
        stride=int(args.stride),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
