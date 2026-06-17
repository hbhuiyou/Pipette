import argparse
import os
from dataclasses import dataclass
from typing import Iterable

import h5py
import schema as h5s


@dataclass
class DemoStats:
    name: str
    steps: int
    dataset_count: int
    group_count: int
    bytes_total: int


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def dataset_nbytes(ds: h5py.Dataset) -> int:
    # Works for regular numeric datasets and most fixed-size string/object arrays.
    try:
        return int(ds.size * ds.dtype.itemsize)
    except Exception:
        return 0


def walk_group(g: h5py.Group) -> Iterable[tuple[str, h5py.Dataset | h5py.Group]]:
    for key in g.keys():
        obj = g[key]
        yield key, obj
        if isinstance(obj, h5py.Group):
            for child_path, child_obj in _walk_group_recursive(obj, prefix=key):
                yield child_path, child_obj


def _walk_group_recursive(g: h5py.Group, prefix: str) -> Iterable[tuple[str, h5py.Dataset | h5py.Group]]:
    for key in g.keys():
        obj = g[key]
        path = f"{prefix}/{key}"
        yield path, obj
        if isinstance(obj, h5py.Group):
            for child_path, child_obj in _walk_group_recursive(obj, prefix=path):
                yield child_path, child_obj


def compute_group_stats(g: h5py.Group) -> tuple[int, int, int]:
    dataset_count = 0
    group_count = 0
    bytes_total = 0
    for _, obj in walk_group(g):
        if isinstance(obj, h5py.Dataset):
            dataset_count += 1
            bytes_total += dataset_nbytes(obj)
        elif isinstance(obj, h5py.Group):
            group_count += 1
    return dataset_count, group_count, bytes_total


def infer_steps(demo_group: h5py.Group) -> int:
    # Prefer actions length if available, else max first-dim among datasets.
    if h5s.ACTIONS in demo_group and isinstance(demo_group[h5s.ACTIONS], h5py.Dataset):
        actions_ds = demo_group[h5s.ACTIONS]
        if actions_ds.ndim >= 1:
            return int(actions_ds.shape[0])

    step_candidates: list[int] = []
    for _, obj in walk_group(demo_group):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 1:
            step_candidates.append(int(obj.shape[0]))

    return max(step_candidates) if step_candidates else 0


def print_tree(g: h5py.Group, indent: str = "") -> None:
    keys = list(g.keys())
    for i, key in enumerate(keys):
        obj = g[key]
        connector = "└─" if i == len(keys) - 1 else "├─"
        if isinstance(obj, h5py.Dataset):
            shape = "x".join(str(x) for x in obj.shape)
            print(f"{indent}{connector} {key} [Dataset shape={shape}, dtype={obj.dtype}]")
        else:
            print(f"{indent}{connector} {key}/")
            extension = "   " if i == len(keys) - 1 else "│  "
            print_tree(obj, indent + extension)


def resolve_demo_root(h5f: h5py.File) -> h5py.Group:
    return h5s.resolve_demo_root(h5f)


def find_demo_groups(root: h5py.Group) -> list[str]:
    return h5s.sorted_demo_names(root)


def try_read_env_name(h5f: h5py.File) -> str | None:
    # Most IsaacLab exports keep env_name in file attrs.
    value = h5f.attrs.get("env_name", None)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect IsaacLab-style HDF5 demo dataset")
    parser.add_argument("--file", type=str, required=True, help="Path to .hdf5 file")
    parser.add_argument("--show-tree", action="store_true", help="Print HDF5 hierarchy tree")
    parser.add_argument("--show-attrs", action="store_true", help="Print file-level attributes")
    parser.add_argument("--max-demos", type=int, default=0, help="Limit per-demo rows printed (0 means all)")
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    with h5py.File(file_path, "r") as h5f:
        env_name = try_read_env_name(h5f)
        root = resolve_demo_root(h5f)
        demo_names = find_demo_groups(root)

        print("=" * 72)
        print("HDF5 Dataset Summary")
        print("=" * 72)
        print(f"File: {file_path}")
        print(f"File size: {human_bytes(os.path.getsize(file_path))}")
        print(f"Env name: {env_name if env_name is not None else '<missing>'}")
        print(f"Demo root: /{root.name.lstrip('/')}")
        print(f"Demo count: {len(demo_names)}")

        if args.show_attrs:
            print("\n[File Attributes]")
            if len(h5f.attrs) == 0:
                print("  <none>")
            else:
                for k in h5f.attrs.keys():
                    print(f"  {k}: {h5f.attrs[k]}")

        if args.show_tree:
            print("\n[HDF5 Tree]")
            print(f"{root.name}/")
            print_tree(root)

        if len(demo_names) == 0:
            print("\nNo demo_* groups found under demo root.")
            return

        demo_stats: list[DemoStats] = []
        for name in demo_names:
            group = root[name]
            if not isinstance(group, h5py.Group):
                continue
            steps = infer_steps(group)
            dataset_count, group_count, bytes_total = compute_group_stats(group)
            demo_stats.append(
                DemoStats(
                    name=name,
                    steps=steps,
                    dataset_count=dataset_count,
                    group_count=group_count,
                    bytes_total=bytes_total,
                )
            )

        total_steps = sum(s.steps for s in demo_stats)
        total_bytes = sum(s.bytes_total for s in demo_stats)
        mean_steps = (total_steps / len(demo_stats)) if demo_stats else 0.0

        print("\n[Aggregate]")
        print(f"Total steps (estimated): {total_steps}")
        print(f"Mean steps per demo: {mean_steps:.2f}")
        print(f"Data bytes across demos (estimated): {human_bytes(total_bytes)}")

        print("\n[Per Demo]")
        print(f"{'demo':<12} {'steps':>10} {'datasets':>10} {'groups':>8} {'size':>14}")
        print("-" * 72)

        rows = demo_stats
        if args.max_demos > 0:
            rows = demo_stats[: args.max_demos]

        for s in rows:
            print(
                f"{s.name:<12} {s.steps:>10} {s.dataset_count:>10} "
                f"{s.group_count:>8} {human_bytes(s.bytes_total):>14}"
            )

        if args.max_demos > 0 and len(demo_stats) > args.max_demos:
            print(f"... ({len(demo_stats) - args.max_demos} demos omitted)")


if __name__ == "__main__":
    main()
