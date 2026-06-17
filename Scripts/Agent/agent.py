from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_client import OpenAICompatibleClient, fallback_intent
from script_catalog import (
    ACTIONS,
    PROJECT_ROOT,
    SCRIPTS_DIR,
    SERVER_SCRIPT,
    action_descriptions,
    default_task_id,
    get_task,
    inference_script_for,
    list_tasks,
    reload_task_registry,
)
from terminal_launcher import format_command, launch_new_terminal


INFERENCE_TYPES = ["act", "smolvla", "pi0"]
TRAIN_POLICY_TYPES = ["act", "smolvla", "pi0"]
TASK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass
class ActiveJob:
    title: str
    argv: list[str]
    process: subprocess.Popen
    pid_file: Path


ACTIVE_JOB: ActiveJob | None = None


class AgentCancel(Exception):
    pass


def train_defaults(policy_type: str) -> tuple[int, int]:
    if policy_type == "act":
        return 32, 15000
    if policy_type == "smolvla":
        return 8, 13000
    if policy_type == "pi0":
        return 4, 20000
    return 8, 20000


def bool_default(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def load_local_config() -> None:
    config_path = Path(__file__).resolve().parent / "local_config.env"
    if not config_path.exists():
        return
    for line_number, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            print(f"[WARN] 跳过 local_config.env 第 {line_number} 行：缺少 '='。")
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = strip_env_value(value.strip())
        if not key:
            print(f"[WARN] 跳过 local_config.env 第 {line_number} 行：变量名为空。")
            continue
        os.environ.setdefault(key, value)


def strip_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def main() -> None:
    load_local_config()
    print("=== Completed Script Agent ===")
    print("你可以直接说：我要采集数据、搭建环境、注册任务、运行 act 推理、运行 pi0 推理、查看数据集、转换 LeRobot。")
    print("主界面输入 exit / 退出 会结束 Agent；流程中输入 exit 会返回主界面。\n")

    llm = OpenAICompatibleClient()
    if llm.available:
        print(f"[INFO] 已启用 API 意图理解，模型: {llm.model}")
    else:
        print("[INFO] 未检测到 OPENAI_API_KEY，将使用本地规则理解自然语言。")

    while True:
        try:
            user_text = input("\n你想做什么？> ").strip()
        except EOFError:
            print("\n输入结束，已退出。")
            return
        if not user_text:
            continue

        if is_stop_active_job_request(user_text):
            stop_active_job()
            continue

        local_intent = fallback_intent(user_text)
        if local_intent.action == "exit":
            intent = local_intent
        elif llm.available:
            intent = llm.classify_intent(
                user_text=user_text,
                actions=action_descriptions(),
                tasks=[task.__dict__ for task in list_tasks()],
            )
        else:
            intent = local_intent

        if llm.available and (intent.source != "api" or intent.confidence < 0.45):
            fallback = local_intent
            if fallback.confidence >= intent.confidence:
                fallback.reply = intent.reply or fallback.reply
                intent = fallback

        if intent.reply:
            print(f"[Agent] {intent.reply}")

        if intent.action == "exit":
            print("好的，已退出。")
            return
        if intent.action == "stop_active_job":
            stop_active_job()
            continue

        handler = HANDLERS.get(intent.action, show_help)
        try:
            handler(intent.parameters)
        except AgentCancel:
            print("已取消当前操作，回到主界面。")
            continue
        except KeyboardInterrupt:
            print("\n已取消当前操作。")
        except Exception as exc:
            print(f"[ERROR] 操作失败: {exc}")


def show_help(_: dict[str, Any] | None = None) -> None:
    print("\n我可以帮你分发这些脚本：")
    for action in ACTIONS.values():
        if action.name == "help":
            continue
        print(f"  - {action.title}: {action.description}")
    print("\n常用说法示例：")
    print("  我要采集数据")
    print("  搭建环境")
    print("  运行 act 推理")
    print("  训练 act 模型")
    print("  注册一个新任务")
    print("  用 pi0 运行推理评估")
    print("  查看 pick_up_the_tube 数据集")
    print("  把 hdf5 转成 lerobot")


def handle_collect_data(params: dict[str, Any]) -> None:
    task = choose_task(params.get("task_id"))
    num_demos = ask_int("要采集多少条 demo？", default=int(params.get("num_demos") or 5), minimum=1)
    headless = ask_yes_no("是否无界面运行？采集通常需要界面，所以默认否。", default=False)

    argv = python_cmd(
        ACTIONS["collect_data"].script,
        ["--task_id", task.task_id, "--num_demos", str(num_demos)] + (["--headless"] if headless else []),
    )
    print_collection_guide(task.task_id, task.dataset_file, num_demos)
    launch(argv, "采集数据")


def handle_augment_data(params: dict[str, Any]) -> None:
    task = choose_task(params.get("task_id"))
    dataset_file = ask_text("输入 HDF5 数据路径", default=str(params.get("dataset_file") or task.dataset_file))
    output_file = ask_text("输出文件路径，留空则自动生成", default="", required=False)
    worker_count = ask_int("并行 worker 数量，建议 2 到 4", default=3, minimum=1, maximum=4)
    light_scales = ask_text("光照增强倍率", default="0.8")
    temporal_scales = ask_text("时间速度倍率", default="1.2")
    camera_jitter_count = ask_int("每个变体的相机扰动数量", default=0, minimum=0)
    include_original = ask_yes_no("是否把原始 demo 也复制到输出里？", default=True)
    headless = ask_yes_no("是否无界面运行？", default=True)

    args = [
        "--task_id",
        task.task_id,
        "--dataset_file",
        dataset_file,
        "--num_envs",
        str(worker_count),
        "--light_intensity_scales",
        light_scales,
        "--temporal_speed_scales",
        temporal_scales,
        "--camera_jitter_count",
        str(camera_jitter_count),
    ]
    if output_file:
        args += ["--output_file", output_file]
    if include_original:
        args.append("--include_original")
    if headless:
        args.append("--headless")

    launch(python_cmd(ACTIONS["augment_data"].script, args), "增强数据")


def handle_replay_data(params: dict[str, Any]) -> None:
    task = choose_task(params.get("task_id"))
    dataset_file = ask_text("要回放的 HDF5 数据路径", default=str(params.get("dataset_file") or task.dataset_file))
    demo_index = ask_int("回放哪个 demo？-1 表示全部", default=-1, minimum=-1)
    replay_mode = ask_choice("回放模式", ["state", "action"], default="state")
    speed = ask_text("回放速度倍率", default="1.0")

    argv = python_cmd(
        ACTIONS["replay_data"].script,
        [
            "--task_id",
            task.task_id,
            "--dataset_file",
            dataset_file,
            "--demo_index",
            str(demo_index),
            "--replay_mode",
            replay_mode,
            "--speed",
            speed,
        ],
    )
    launch(argv, "回放数据")


def handle_inspect_dataset(params: dict[str, Any]) -> None:
    task = choose_task(params.get("task_id"), allow_skip=True)
    default_file = task.dataset_file if task is not None else str(params.get("dataset_file") or "")
    dataset_file = ask_text("要查看的 HDF5 数据路径", default=default_file)
    show_tree = ask_yes_no("是否打印完整 HDF5 树？", default=False)
    show_attrs = ask_yes_no("是否打印文件属性？", default=True)

    args = ["--file", dataset_file]
    if show_tree:
        args.append("--show-tree")
    if show_attrs:
        args.append("--show-attrs")
    launch(python_cmd(ACTIONS["inspect_dataset"].script, args), "查看数据集")


def handle_convert_lerobot(params: dict[str, Any]) -> None:
    task = choose_task(params.get("task_id"), allow_skip=True)
    default_hdf5 = task.dataset_file if task is not None else str(params.get("dataset_file") or "")
    hdf5_path = ask_text("输入 HDF5 数据路径", default=default_hdf5)
    task_id = task.task_id if task is not None else Path(hdf5_path).stem
    output_dir = ask_text("输出 LeRobot 数据集目录（绝对路径）", default=str(params.get("output_dir") or ""))
    output_path = Path(output_dir).expanduser()
    if not (output_path.is_absolute() or output_dir.startswith("/")):
        raise ValueError(f"LeRobot 输出目录必须是绝对路径：{output_dir}")
    hdf5_path = str(resolve_project_path(hdf5_path))
    output_dir = output_dir if output_dir.startswith("/") else str(output_path)
    repo_id = task_id
    fps = 10
    frame_filter = "fresh"
    stride = 3
    convert_python = resolve_lerobot_python()
    convert_cwd = resolve_lerobot_workdir()

    print("\n转换参数已自动设置：")
    print(f"  python: {convert_python}")
    print(f"  workdir: {convert_cwd}")
    print(f"  repo-id: {repo_id}")
    print(f"  fps: {fps}")
    print(f"  frame-filter: {frame_filter}")
    print(f"  stride: {stride}")

    argv = python_cmd(
        ACTIONS["convert_lerobot"].script,
        [
            "--hdf5-path",
            hdf5_path,
            "--output-dir",
            output_dir,
            "--repo-id",
            repo_id,
            "--fps",
            str(fps),
            "--frame-filter",
            frame_filter,
            "--stride",
            str(stride),
        ],
        python_exe=convert_python,
    )
    launch(
        argv,
        "转换 LeRobot",
        cwd=convert_cwd,
        keep_open_default=True,
        clean_python_env=True,
    )


def handle_train_model(params: dict[str, Any]) -> None:
    task = choose_task(params.get("task_id"))
    task_slug = train_task_slug(task.task_id)
    policy_type = ask_choice("训练模型类型", TRAIN_POLICY_TYPES, default=str(params.get("policy_type") or "act"))

    dataset_repo_id = ask_text("LeRobot 数据集目录（绝对路径）", default=str(params.get("dataset_repo_id") or ""))
    dataset_repo_path = Path(dataset_repo_id).expanduser()
    if not (dataset_repo_path.is_absolute() or dataset_repo_id.startswith("/")):
        raise ValueError(f"LeRobot 数据集目录必须是绝对路径：{dataset_repo_id}")
    dataset_repo_id = dataset_repo_id if dataset_repo_id.startswith("/") else str(dataset_repo_path)

    default_policy_repo = f"local/{policy_type}_{task_slug}"
    policy_repo_id = default_policy_repo

    model_root = resolve_project_path(os.getenv("LEROBOT_MODEL_ROOT", "models"))
    default_output_dir = str(model_root / f"{task_slug}_{policy_type}")
    output_dir = ask_text("模型输出目录", default=str(params.get("output_dir") or default_output_dir))

    default_batch_size, default_steps = train_defaults(policy_type)
    batch_size = ask_int("batch size", default=int(params.get("batch_size") or default_batch_size), minimum=1)
    steps = ask_int("训练步数 steps", default=int(params.get("steps") or default_steps), minimum=1)
    wandb_enable = ask_yes_no("是否启用 wandb", default=bool_default(params.get("wandb_enable"), False))
    push_to_hub = ask_yes_no("是否上传到 Hugging Face Hub", default=bool_default(params.get("push_to_hub"), False))

    train_cmd = resolve_lerobot_train_command()
    train_cwd = resolve_lerobot_workdir()
    argv = [
        train_cmd,
        f"--dataset.repo_id={dataset_repo_id}",
        f"--policy.type={policy_type}",
        f"--policy.repo_id={policy_repo_id}",
        f"--output_dir={output_dir}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--wandb.enable={str(wandb_enable).lower()}",
        f"--policy.push_to_hub={str(push_to_hub).lower()}",
    ]

    print("\n训练参数已设置：")
    print(f"  command: {train_cmd}")
    print(f"  workdir: {train_cwd}")
    print(f"  dataset.repo_id: {dataset_repo_id}")
    print(f"  policy.type: {policy_type}")
    print(f"  policy.repo_id: {policy_repo_id}")
    print(f"  output_dir: {output_dir}")
    print(f"  batch_size: {batch_size}")
    print(f"  steps: {steps}")
    print(f"  wandb.enable: {str(wandb_enable).lower()}")
    print(f"  policy.push_to_hub: {str(push_to_hub).lower()}")

    launch(
        argv,
        "训练模型",
        cwd=train_cwd,
        keep_open_default=True,
        clean_python_env=True,
    )


def handle_delete_demo(params: dict[str, Any]) -> None:
    task = choose_task(params.get("task_id"), allow_skip=True)
    default_file = task.dataset_file if task is not None else str(params.get("dataset_file") or "")
    dataset_file = ask_text("要删除 demo 的 HDF5 文件", default=default_file)
    indices = ask_text("要删除的 demo 编号，用逗号分隔，例如 0,2,5")
    dry_run = ask_yes_no("先 dry-run 预览，不实际删除？强烈建议默认是。", default=True)

    args = ["--file", dataset_file, "--indices", indices]
    if dry_run:
        args.append("--dry-run")
    launch(python_cmd(ACTIONS["delete_demo"].script, args), "删除 demo")


def handle_register_task(params: dict[str, Any]) -> None:
    print("\n我会按固定模板注册任务。你只需要提供 USD 场景路径和任务描述。")
    print("task_id 和 language_instruction 会由大模型根据任务描述自动生成。")
    print("可参考约束文件：Agent/task_registration_rules.md")

    usd_path = ask_usd_path(str(params.get("usd_path") or ""))
    user_description = ask_text("任务描述，例如：让机械臂拿起蓝色试管", default=str(params.get("description") or ""))
    draft = generate_task_registration_draft(user_description)
    if draft.get("warning"):
        print(f"[WARN] {draft['warning']}")

    task_id = unique_task_id(sanitize_task_id(draft.get("task_id") or "custom_task"))
    description = draft.get("description") or user_description
    language_instruction = draft.get("language_instruction") or task_id
    dataset_file = f"./datasets/{task_id}.hdf5"

    print("\n请确认要写入 task_registry 的内容：")
    print(f"  task_id: {task_id}")
    print(f"  usd_path: {usd_path}")
    print(f"  description: {description}")
    print(f"  language_instruction: {language_instruction}")
    print(f"  dataset_file: {dataset_file}  （自动按 ./datasets/任务id.hdf5 生成）")
    print("  其余字段：沿用 make_task 默认值")
    if not ask_yes_no("确认注册这个任务？", default=False):
        print("已取消注册任务。")
        return

    register_task_in_registry(
        task_id=task_id,
        usd_path=usd_path,
        description=description,
        language_instruction=language_instruction,
        dataset_file=dataset_file,
    )
    reload_task_registry()
    print(f"已注册任务：{task_id}")
    print("之后输入“采集数据”或“转换 LeRobot”时，这个任务会出现在任务列表中。")


def handle_build_environment(params: dict[str, Any]) -> None:
    default_usd = resolve_project_path(os.getenv("AGENT_ENV_TEMPLATE_USD", "Asset/lab.usd"))
    usd_path = str(resolve_project_path(str(params.get("usd_path") or default_usd)))
    default_asset_dir = resolve_project_path(os.getenv("AGENT_ASSET_DIR", "Asset"))
    asset_dir = str(resolve_project_path(str(params.get("asset_dir") or default_asset_dir)))

    print("\n将打开预设 USD 场景用于搭建/修改环境：")
    print(f"  {usd_path}")
    print("Content Browser 将默认定位到资源文件夹：")
    print(f"  {asset_dir}")
    print("请在 Isaac Sim 中修改场景。完成后使用 File > Save As 保存为新的 USD 文件。")
    print("保存后的 USD 路径可以在之后“注册任务”时填写。")

    launch(
        python_cmd(ACTIONS["build_environment"].script, ["--usd-path", usd_path, "--asset-dir", asset_dir]),
        "搭建环境",
        keep_open_default=True,
    )


def handle_run_inference(params: dict[str, Any]) -> None:
    task = choose_task(params.get("task_id"))
    policy_type = ask_choice("模型/客户端策略类型", INFERENCE_TYPES, default=str(params.get("policy_type") or "act"))
    policy_path = ask_text("模型路径或 Hugging Face repo id", default=str(params.get("policy_path") or ""))
    endpoint = "tcp://127.0.0.1:5555"
    episodes = ask_int("评估回合数", default=int(params.get("episodes") or 100), minimum=1)
    gui = ask_yes_no("是否有界面运行？", default=True)
    headless = not gui

    server_args = [
        "--policy-path",
        policy_path,
        "--policy-type",
        policy_type,
        "--bind",
        endpoint,
        "--device",
        "cuda",
    ]
    print("\n将先自动启动模型服务：")
    print(f"  policy-path: {policy_path}")
    print(f"  policy-type: {policy_type}")
    print(f"  bind: {endpoint}")
    server_started = launch(
        python_cmd(SERVER_SCRIPT, server_args, python_exe=resolve_lerobot_python()),
        "模型服务",
        cwd=resolve_lerobot_workdir(),
        keep_open_default=True,
        clean_python_env=True,
    )
    if not server_started:
        print("模型服务未启动，已取消本次推理。")
        return

    script = inference_script_for(policy_type)
    args = ["--task-id", task.task_id, "--server-endpoint", endpoint]
    if policy_type == "act":
        args += ["--episodes", str(max(1, int(episodes)))]
    elif policy_type == "pi0":
        args += ["--episodes", str(max(1, int(episodes)))]
    else:
        args += ["--episodes", str(max(1, int(episodes)))]
    if headless:
        args.append("--headless")

    wait_seconds = ask_int("模型服务预热等待秒数", default=int(os.getenv("INFERENCE_SERVER_WAIT_SEC", "10")), minimum=0)
    if wait_seconds > 0:
        print(f"等待模型服务加载 {wait_seconds} 秒...")
        time.sleep(float(wait_seconds))

    print("\n即将启动 IsaacSim 推理客户端。")
    launch(python_cmd(script, args), "运行推理")


def generate_task_registration_draft(user_description: str) -> dict[str, str]:
    llm = OpenAICompatibleClient()
    draft = llm.generate_task_registration_fields(user_description)
    task_id = sanitize_task_id(draft.get("task_id") or "")
    if not task_id:
        task_id = sanitize_task_id(user_description)
    if not task_id:
        task_id = "custom_task"

    language_instruction = sanitize_task_id(draft.get("language_instruction") or "")
    if not language_instruction:
        language_instruction = task_id

    description = str(draft.get("description") or user_description).strip()
    return {
        "task_id": task_id,
        "description": description,
        "language_instruction": language_instruction,
        "source": str(draft.get("source") or "fallback"),
        "warning": str(draft.get("warning") or ""),
    }


def sanitize_task_id(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return ""
    if not normalized[0].isalpha():
        normalized = f"task_{normalized}"
    if not TASK_ID_PATTERN.fullmatch(normalized):
        return ""
    return normalized


def unique_task_id(task_id: str) -> str:
    existing = {task.task_id for task in list_tasks()}
    if task_id not in existing:
        return task_id
    suffix = 2
    while f"{task_id}_{suffix}" in existing:
        suffix += 1
    return f"{task_id}_{suffix}"


def task_constant_name(task_id: str) -> str:
    return f"{task_id.upper()}_TASK_ID"


def py_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def project_relative_path(path_text: str) -> str:
    path = resolve_project_path(path_text)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def ask_usd_path(default: str = "") -> str:
    while True:
        usd_path = ask_text("搭建好的 USD 场景路径", default=default)
        if not usd_path.lower().endswith((".usd", ".usda", ".usdc")):
            print("USD 场景路径应以 .usd、.usda 或 .usdc 结尾。")
            continue
        return usd_path


def register_task_in_registry(
    *,
    task_id: str,
    usd_path: str,
    description: str,
    language_instruction: str,
    dataset_file: str,
) -> None:
    usd_path = project_relative_path(usd_path)
    registry_path = SCRIPTS_DIR / "Data" / "task_registry.py"
    text = registry_path.read_text(encoding="utf-8")
    if f'"{task_id}"' in text or f"'{task_id}'" in text:
        raise ValueError(f"task_registry 中已经存在 {task_id}")
    constant_name = task_constant_name(task_id)
    if re.search(rf"^\s*{re.escape(constant_name)}\s*=", text, flags=re.M):
        raise ValueError(f"task_registry 中已经存在常量 {constant_name}")

    constants_marker = "\n\n\nDEFAULT_FRANKA_JOINT_POS"
    constants_insert_at = text.find(constants_marker)
    if constants_insert_at == -1:
        raise RuntimeError("没有找到任务常量区结束位置，已取消写入。")
    constant_entry = f"{constant_name} = {py_string(task_id)}"

    entry = (
        f"    {constant_name}: make_task(\n"
        f"        task_id={constant_name},\n"
        f"        description={py_string(description)},\n"
        f"        usd_path={py_string(usd_path)},\n"
        f"        dataset_file={py_string(dataset_file)},\n"
        f"        language_instruction={py_string(language_instruction)},\n"
        f"    ),\n"
    )
    presets_marker = "\n}\n\n\ndef get_task_preset"
    presets_insert_at = text.find(presets_marker)
    if presets_insert_at == -1:
        raise RuntimeError("没有找到 TASK_PRESETS 的结束位置，已取消写入。")

    text = text[:constants_insert_at] + "\n" + constant_entry + text[constants_insert_at:]
    presets_insert_at = text.find(presets_marker)
    registry_path.write_text(text[:presets_insert_at] + entry + text[presets_insert_at:], encoding="utf-8")


def choose_task(preferred: Any = None, allow_skip: bool = False):
    tasks = list_tasks()
    if preferred:
        preferred_str = str(preferred)
        for task in tasks:
            if task.task_id == preferred_str:
                return task

    print("\n可选任务：")
    if allow_skip:
        print("  0. 不指定任务，手动输入路径")
    for index, task in enumerate(tasks, start=1):
        print(f"  {index}. {task.task_id}  [{task.dataset_file}]")
        print(f"     {task.description}")

    default_id = default_task_id()
    while True:
        raw = input(f"请选择任务编号或 task_id，直接回车默认 {default_id}> ").strip()
        if is_exit_request(raw):
            raise AgentCancel()
        if not raw:
            return get_task(default_id)
        if allow_skip and raw == "0":
            return None
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(tasks):
                return tasks[index - 1]
        for task in tasks:
            if raw == task.task_id:
                return task
        print("没有找到这个任务，请重新输入。")


def ask_text(prompt: str, default: str = "", required: bool = True) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        raw = input(f"{prompt}{suffix}> ").strip()
        if is_exit_request(raw):
            raise AgentCancel()
        value = raw or default
        if value or not required:
            return value
        print("这里需要一个值。")


def ask_int(prompt: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]> ").strip()
        if is_exit_request(raw):
            raise AgentCancel()
        value_text = raw or str(default)
        try:
            value = int(value_text)
        except ValueError:
            print("请输入整数。")
            continue
        if minimum is not None and value < minimum:
            print(f"不能小于 {minimum}。")
            continue
        if maximum is not None and value > maximum:
            print(f"不能大于 {maximum}。")
            continue
        return value


def ask_yes_no(prompt: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{hint}]> ").strip().lower()
        if is_exit_request(raw):
            raise AgentCancel()
        if not raw:
            return default
        if raw in {"y", "yes", "是", "对", "1"}:
            return True
        if raw in {"n", "no", "否", "不", "0"}:
            return False
        print("请输入 y 或 n。")


def ask_choice(prompt: str, choices: list[str], default: str) -> str:
    default = default if default in choices else choices[0]
    print(f"\n{prompt}：")
    for index, item in enumerate(choices, start=1):
        marker = " (默认)" if item == default else ""
        print(f"  {index}. {item}{marker}")
    while True:
        raw = input(f"请选择编号或名称 [{default}]> ").strip().lower()
        if is_exit_request(raw):
            raise AgentCancel()
        if not raw:
            return default
        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(choices):
                return choices[index - 1]
        if raw in choices:
            return raw
        print("选择无效，请重新输入。")


def python_cmd(script: Path | None, args: list[str], python_exe: str | None = None) -> list[str]:
    if script is None:
        raise ValueError("缺少脚本路径。")
    python_exe = python_exe or os.getenv("AGENT_PYTHON", sys.executable)
    return [python_exe, str(script), *args]


def launch(
    argv: list[str],
    title: str,
    cwd: Path = PROJECT_ROOT,
    keep_open_default: bool = False,
    clean_python_env: bool = False,
) -> bool:
    global ACTIVE_JOB
    print("\n即将在新终端运行：")
    print(f"  {format_command(argv)}")
    if not ask_yes_no("确认启动？", default=True):
        print("已取消启动。")
        return False
    runtime_dir = Path(__file__).resolve().parent / ".runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    pid_file = runtime_dir / "active_job.pid"
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass
    env_unset = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE") if clean_python_env else ()
    process = launch_new_terminal(
        argv,
        cwd=cwd,
        title=title,
        pid_file=pid_file,
        keep_open_default=keep_open_default,
        env_unset=env_unset,
    )
    ACTIVE_JOB = ActiveJob(title=title, argv=list(argv), process=process, pid_file=pid_file)
    print("已创建新终端。你可以继续在这里输入下一条自然语言指令。")
    print("如果要结束这个任务，请在主界面输入：停止当前任务。")
    return True


def stop_active_job() -> None:
    global ACTIVE_JOB
    if ACTIVE_JOB is None:
        print("当前没有由 Agent 记录的运行任务。")
        return

    job = ACTIVE_JOB
    pid = wait_for_pid_file(job.pid_file)
    if pid is None:
        pid = job.process.pid if job.process.poll() is None else None
    if pid is None:
        print(f"任务「{job.title}」似乎已经结束。")
        ACTIVE_JOB = None
        return

    print(f"正在停止当前任务「{job.title}」...")
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            terminate_pid(pid, signal.SIGTERM)
            time.sleep(1.0)
            if process_exists(pid):
                terminate_pid(pid, signal.SIGKILL)
    finally:
        ACTIVE_JOB = None
    print("已发送停止信号。若 IsaacSim 窗口仍未关闭，请在采集窗口按 X 或手动关闭。")


def read_pid_file(pid_file: Path) -> int | None:
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def wait_for_pid_file(pid_file: Path, timeout_seconds: float = 2.0) -> int | None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        pid = read_pid_file(pid_file)
        if pid is not None:
            return pid
        time.sleep(0.05)
    return read_pid_file(pid_file)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_lerobot_python() -> str:
    configured = os.getenv("LEROBOT_PYTHON", "").strip()
    if configured:
        return str(resolve_project_path(configured))
    return os.getenv("AGENT_PYTHON", sys.executable)


def resolve_lerobot_workdir() -> Path:
    configured = os.getenv("LEROBOT_WORKDIR", "").strip()
    return resolve_project_path(configured) if configured else PROJECT_ROOT


def resolve_lerobot_train_command() -> str:
    configured = os.getenv("LEROBOT_TRAIN", "").strip()
    if configured:
        return str(resolve_project_path(configured))
    lerobot_python = Path(resolve_lerobot_python()).expanduser()
    candidate = lerobot_python.parent / ("lerobot-train.exe" if os.name == "nt" else "lerobot-train")
    if candidate.exists():
        return str(candidate)
    return "lerobot-train"


def train_task_slug(task_id: str) -> str:
    value = task_id.strip().lower()
    replacements = {
        "pick_up_the_tube": "pick_up_the_tube",
        "pick_up_the_pipette": "pick_pipette",
        "open_the_centrifuge_lid": "open_centrifuge_lid",
        "close_the_centrifuge_lid": "close_centrifuge_lid",
        "take_out_the_petri_dish": "take_out_petri_dish",
        "place_the_petri_dish": "place_petri_dish",
    }
    if value in replacements:
        return replacements[value]
    return value.replace("_the_", "_").replace("the_", "")


def terminate_pid(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
        return
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def is_exit_request(text: str) -> bool:
    value = text.strip().lower()
    if not value:
        return False
    return value in {
        "exit",
        "quit",
        "q",
        "退出",
        "结束",
        "取消",
        "返回",
        "关闭",
        "退出脚本",
        "关闭脚本",
        "结束脚本",
        "采集完成，退出脚本",
        "采集完成,退出脚本",
    }


def is_stop_active_job_request(text: str) -> bool:
    value = text.strip().lower()
    if not value:
        return False
    keywords = {
        "采集完成",
        "完成采集",
        "结束采集",
        "停止采集",
        "停止当前任务",
        "结束当前任务",
        "关闭当前任务",
        "停止isaacsim",
        "关闭isaacsim",
        "stop collection",
        "stop current job",
    }
    return value in keywords or any(keyword in value for keyword in keywords)


def print_collection_guide(task_id: str, dataset_file: str, num_demos: int) -> None:
    print("\n采集说明：")
    print(f"  任务: {task_id}")
    print(f"  目标条数: {num_demos}")
    print(f"  数据会追加/保存到: {dataset_file}")
    print('  新窗口中 IsaacLab 就绪后，按 "R" 保存当前 demo 并进入下一条。')
    print('  按 "SPACE" 跳过当前 demo，按 "P" 退出采集。')
    print("  采集时请让 Isaac Sim/IsaacLab 窗口保持焦点，便于键盘控制生效。")


HANDLERS = {
    "collect_data": handle_collect_data,
    "augment_data": handle_augment_data,
    "replay_data": handle_replay_data,
    "inspect_dataset": handle_inspect_dataset,
    "convert_lerobot": handle_convert_lerobot,
    "train_model": handle_train_model,
    "delete_demo": handle_delete_demo,
    "register_task": handle_register_task,
    "build_environment": handle_build_environment,
    "run_inference": handle_run_inference,
    "help": show_help,
}


if __name__ == "__main__":
    main()
