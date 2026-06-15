import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


PIPETTE_ROOT = Path(__file__).resolve().parents[2]


def resolve_pipette_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else (PIPETTE_ROOT / path).resolve()


def focus_content_browser(asset_dir: str) -> bool:
    asset_dir = str(resolve_pipette_path(asset_dir))

    try:
        import carb.settings

        settings = carb.settings.get_settings()
        for key in (
            "/persistent/app/content_browser/default_dir",
            "/persistent/exts/omni.kit.window.content_browser/default_dir",
            "/persistent/exts/omni.kit.browser.asset/root_folder",
        ):
            settings.set(key, asset_dir)
    except Exception as exc:
        print(f"[WARN] Could not write Content Browser settings: {exc}")

    print("[INFO] Asset directory for dragging assets:")
    print(f"  {asset_dir}")
    print("[INFO] Content Browser auto-focus is kept conservative to avoid Isaac Sim 5.1 UI overlap.")
    print("[INFO] If needed, open this path manually from My Computer.")
    return False


def main():
    parser = argparse.ArgumentParser(description="Open a USD scene for manual environment editing.")
    parser.add_argument(
        "--usd-path",
        type=str,
        default="Asset/lab.usd",
        help="USD scene to open for editing.",
    )
    parser.add_argument(
        "--asset-dir",
        type=str,
        default="Asset",
        help="Directory to show in the Isaac Sim Content Browser.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Environment editing requires an interactive Isaac Sim window.
    args_cli.headless = False

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import omni.usd

    usd_path = str(resolve_pipette_path(args_cli.usd_path))
    print("[INFO] Opening USD scene for environment editing:")
    print(f"  {usd_path}")
    print("[INFO] Please edit the scene in Isaac Sim, then use File > Save or Save As.")
    print("[INFO] The saved USD path can be used later when registering a task.")

    context = omni.usd.get_context()
    opened = context.open_stage(usd_path)
    if opened is False:
        raise RuntimeError(f"Failed to open USD scene: {usd_path}")

    # Give Isaac Sim 5.1 UI extensions a moment to finish creating their windows.
    for _ in range(30):
        simulation_app.update()

    focus_content_browser(args_cli.asset_dir)

    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
