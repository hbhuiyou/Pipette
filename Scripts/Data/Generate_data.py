import argparse
import os
import subprocess
import sys
from pathlib import Path

import h5py
from isaaclab.app import AppLauncher


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
PIPETTE_ROOT = SCRIPTS_DIR.parent
IMPORT_PATH_CANDIDATES = [
    SCRIPTS_DIR / "Data",
]
for candidate in IMPORT_PATH_CANDIDATES:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from task_registry import DEFAULT_TASK_ID, get_task_preset  # noqa: E402
import schema as ds  # noqa: E402


_DEMO_SORT_FALLBACK = 10**12


def _demo_sort_key(name: str) -> int:
    return ds.demo_sort_key(name)


def _resolve_output_file(input_file: str, output_file: str) -> str:
    if output_file:
        return output_file
    root, ext = os.path.splitext(input_file)
    if ext.lower() not in {".h5", ".hdf5"}:
        ext = ".hdf5"
    return f"{root}_aug_light_multi{ext}"


def _next_demo_index(data_group: h5py.Group) -> int:
    return ds.next_demo_index(data_group)


def _collect_demo_indices(data_group: h5py.Group) -> list[int]:
    sorted_names = ds.sorted_demo_names(data_group)
    demo_indices: list[int] = []
    for name in sorted_names:
        index = _demo_sort_key(name)
        if index < _DEMO_SORT_FALLBACK:
            demo_indices.append(index)
    return demo_indices


def _split_indices(indices: list[int], chunks: int) -> list[list[int]]:
    chunks = max(1, int(chunks))
    if not indices:
        return []
    out: list[list[int]] = [[] for _ in range(chunks)]
    for idx, value in enumerate(indices):
        out[idx % chunks].append(int(value))
    return [x for x in out if x]


def _merge_outputs(chunk_files: list[str], merged_output: str):
    if not chunk_files:
        raise RuntimeError("No chunk output files to merge.")

    total_src_demos = 0
    chunk_demo_counts: dict[str, int] = {}
    for chunk_file in chunk_files:
        count = 0
        with h5py.File(chunk_file, "r") as src_h5:
            if ds.DATA_GROUP in src_h5:
                count = int(len(src_h5[ds.DATA_GROUP].keys()))
        chunk_demo_counts[chunk_file] = count
        total_src_demos += count

    print(
        f"[INFO] Merge scan completed: chunks={len(chunk_files)}, total_input_demos={total_src_demos}",
        flush=True,
    )

    first_file = chunk_files[0]
    out_exists = os.path.exists(merged_output)
    out_mode = "a" if out_exists else "w"
    with h5py.File(first_file, "r") as first_h5, h5py.File(merged_output, out_mode) as out_h5:
        if not out_exists:
            for attr_key, attr_value in first_h5.attrs.items():
                out_h5.attrs[attr_key] = attr_value

        out_h5.attrs["augmentation_pipeline"] = "multi_process_orchestrator"
        out_h5.attrs["augmentation_chunk_count"] = int(len(chunk_files))

        out_data = ds.require_or_create_data_group(out_h5)

        out_demo_id = _next_demo_index(out_data)
        merged_demo_count = 0
        if out_exists:
            print(f"[INFO] Merge append mode: existing output found at {merged_output}", flush=True)
        else:
            print(f"[INFO] Merge create mode: creating {merged_output}", flush=True)
        print(f"[INFO] Merge output demo id starts from: demo_{out_demo_id}", flush=True)

        for chunk_index, chunk_file in enumerate(chunk_files):
            print(
                f"[INFO] Merging chunk {chunk_index + 1}/{len(chunk_files)}: "
                f"{chunk_file} (demos={chunk_demo_counts.get(chunk_file, 0)})",
                flush=True,
            )
            with h5py.File(chunk_file, "r") as src_h5:
                if ds.DATA_GROUP not in src_h5:
                    print(f"[WARN] Chunk has no '{ds.DATA_GROUP}' group: {chunk_file}", flush=True)
                    continue
                src_data = ds.require_data_group(src_h5)
                demo_names = ds.sorted_demo_names(src_data)
                for src_demo_name in demo_names:
                    src_demo = src_data[src_demo_name]
                    dst_demo_name = ds.demo_name(out_demo_id)
                    merged_demo_count += 1
                    print(
                        f"[INFO] Merge demo {merged_demo_count}/{total_src_demos}: "
                        f"{src_demo_name} -> {dst_demo_name}",
                        flush=True,
                    )
                    out_demo_id += 1
                    src_h5.copy(src_demo, out_data, name=dst_demo_name)

        out_h5.attrs["total"] = int(len(out_data.keys()))
        out_h5.flush()
        print(f"[INFO] Merge finished: output_demos={int(len(out_data.keys()))}", flush=True)


def _run_worker(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    result = subprocess.run(cmd, env=env)
    return int(result.returncode), " ".join(cmd)


def build_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stable 2-4 worker offline augmentation orchestrator. "
            "Runs multiple Replay_light_augmentation workers on demo shards, then merges outputs."
        )
    )
    parser.add_argument("--dataset_file", type=str, default="", help="Input HDF5 dataset path.")
    parser.add_argument("--output_file", type=str, default="", help="Output merged HDF5 path.")
    parser.add_argument("--task_id", type=str, default=DEFAULT_TASK_ID)
    parser.add_argument("--camera_width", type=int, default=400)
    parser.add_argument("--camera_height", type=int, default=400)
    parser.add_argument("--num_envs", type=int, default=2, help="Worker count (recommended 2~4).")
    parser.add_argument("--demo_index", type=int, default=-1, help="Replay one demo index. -1 means all demos.")
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
        help="Comma-separated temporal speed scales, e.g. '1.0,0.8,1.2'.",
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
            "Number of non-zero candidate camera jitter poses sampled by each worker. "
            "0 disables camera jitter."
        ),
    )
    parser.add_argument(
        "--sequential_workers",
        action="store_true",
        help="Run worker shards sequentially. Default runs all workers in parallel.",
    )
    parser.add_argument(
        "--close_timeout_sec",
        type=float,
        default=10.0,
        help="Timeout in seconds passed to worker simulation close watchdog.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args_cli = build_args()
    task_preset = get_task_preset(args_cli.task_id)
    dataset_file = args_cli.dataset_file or task_preset.dataset_file
    output_file = _resolve_output_file(dataset_file, args_cli.output_file)

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    worker_count = max(1, min(4, int(args_cli.num_envs)))
    if int(args_cli.num_envs) != worker_count:
        print(f"[WARN] Clamped --num_envs from {args_cli.num_envs} to {worker_count} (stable range 1~4).")

    with h5py.File(dataset_file, "r") as src_h5:
        src_data = ds.require_data_group(src_h5)
        demo_names = ds.sorted_demo_names(src_data)
        available_demo_indices = _collect_demo_indices(src_data)

    if args_cli.demo_index >= 0:
        selected_name = ds.demo_name(int(args_cli.demo_index))
        if selected_name not in demo_names:
            raise KeyError(f"{selected_name} not found in dataset")
        selected_indices = [int(args_cli.demo_index)]
    else:
        selected_indices = available_demo_indices

    if not selected_indices:
        raise RuntimeError("No demo indices selected.")

    shard_indices = _split_indices(selected_indices, worker_count)
    if not shard_indices:
        raise RuntimeError("No shards generated for workers.")

    python_exe = sys.executable
    worker_script = str(SCRIPT_DIR / "Enhance.py")
    output_dir = os.path.dirname(os.path.abspath(output_file)) or "."
    os.makedirs(output_dir, exist_ok=True)

    chunk_files: list[str] = []
    run_env = dict(os.environ)

    worker_cmds: list[tuple[int, str, list[str]]] = []
    for shard_id, shard in enumerate(shard_indices):
        shard_csv = ",".join(str(v) for v in shard)
        chunk_file = os.path.join(output_dir, f".tmp_aug_chunk_{shard_id}.hdf5")
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
        chunk_files.append(chunk_file)

        cmd = [
            python_exe,
            worker_script,
            "--dataset_file",
            dataset_file,
            "--output_file",
            chunk_file,
            "--task_id",
            str(args_cli.task_id),
            "--camera_width",
            str(int(args_cli.camera_width)),
            "--camera_height",
            str(int(args_cli.camera_height)),
            "--demo_indices",
            shard_csv,
            "--light_intensity_scales",
            str(args_cli.light_intensity_scales),
            "--temporal_speed_scales",
            str(args_cli.temporal_speed_scales),
            "--camera_jitter_count",
            str(int(args_cli.camera_jitter_count)),
            "--close_timeout_sec",
            str(float(args_cli.close_timeout_sec)),
        ]
        if bool(args_cli.include_original):
            cmd.append("--include_original")

        if bool(getattr(args_cli, "headless", False)):
            cmd.append("--headless")
        if bool(getattr(args_cli, "disable_fabric", False)):
            cmd.append("--disable_fabric")

        # Ensure all workers use same rendering device as parent run.
        if getattr(args_cli, "device", None):
            cmd.extend(["--device", str(args_cli.device)])

        worker_cmds.append((shard_id, shard_csv, cmd))

    if bool(args_cli.sequential_workers):
        for shard_id, shard_csv, cmd in worker_cmds:
            print(
                f"[INFO] Running worker {shard_id + 1}/{len(worker_cmds)} on demos: {shard_csv}",
                flush=True,
            )
            ret, cmd_text = _run_worker(cmd, env=run_env)
            if ret != 0:
                raise RuntimeError(f"Worker failed (code={ret}): {cmd_text}")
    else:
        processes: list[tuple[int, str, str, subprocess.Popen]] = []
        for shard_id, shard_csv, cmd in worker_cmds:
            print(
                f"[INFO] Launching worker {shard_id + 1}/{len(worker_cmds)} on demos: {shard_csv}",
                flush=True,
            )
            proc = subprocess.Popen(cmd, env=run_env)
            processes.append((shard_id, shard_csv, " ".join(cmd), proc))

        worker_failures: list[str] = []
        for shard_id, shard_csv, cmd_text, proc in processes:
            ret = int(proc.wait())
            if ret != 0:
                worker_failures.append(
                    f"worker={shard_id} demos={shard_csv} code={ret} cmd={cmd_text}"
                )

        if worker_failures:
            msg = "\n".join(worker_failures)
            raise RuntimeError(f"One or more workers failed:\n{msg}")

    print(f"[INFO] Merging {len(chunk_files)} worker outputs -> {output_file}")
    _merge_outputs(chunk_files=chunk_files, merged_output=output_file)

    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
        except Exception:
            pass

    print(f"[INFO] Multi-env augmentation completed: {output_file}")


if __name__ == "__main__":
    main()
