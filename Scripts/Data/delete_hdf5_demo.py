import argparse
import os
import h5py

import schema as ds


def find_demo_groups(root: h5py.Group):
    return ds.sorted_demo_names(root)


def resolve_demo_root(h5f: h5py.File) -> h5py.Group:
    return ds.resolve_demo_root(h5f)


def delete_demos(file_path: str, demo_indices: list[int], dry_run: bool = False):
    available_indices = set()

    with h5py.File(file_path, "r") as h5f:
        root = resolve_demo_root(h5f)
        demo_names = find_demo_groups(root)

        for name in demo_names:
            index = ds.demo_sort_key(name)
            if index < 10**12:
                available_indices.add(index)

        print(f"Available demo indices: {sorted(available_indices)}")

    valid_indices = set(demo_indices) & available_indices

    if dry_run:
        print(f"[DRY RUN] Would delete demos: {sorted(valid_indices)}")
        print(f"Remaining demos would be: {sorted(available_indices - valid_indices)}")
        return

    if not valid_indices:
        print("No valid demos to delete.")
        return

    with h5py.File(file_path, "r+") as h5f:
        root = resolve_demo_root(h5f)

        indices_to_delete = sorted(valid_indices, reverse=True)
        for idx in indices_to_delete:
            demo_name = ds.demo_name(idx)
            if demo_name in root:
                del root[demo_name]
                print(f"Deleted {demo_name}")

        remaining_demos = ds.sorted_demo_names(root)

        for new_idx, old_name in enumerate(remaining_demos):
            new_name = ds.demo_name(new_idx)
            if old_name != new_name:
                root.move(old_name, new_name)
                print(f"Renamed {old_name} -> {new_name}")

    print(f"Done. Modified: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Delete specified demo rows from HDF5 file (in-place)")
    parser.add_argument("--file", type=str, required=True, help="HDF5 file path")
    parser.add_argument("--indices", type=str, required=True, help="Comma-separated demo indices to delete (e.g., '0,2,5')")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without making changes")
    args = parser.parse_args()

    file_path = os.path.abspath(args.file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    demo_indices = [int(x.strip()) for x in args.indices.split(",")]

    delete_demos(file_path, demo_indices, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
