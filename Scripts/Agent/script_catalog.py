from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = AGENT_DIR.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
DATA_DIR = SCRIPTS_DIR / "Data"
CLIENT_DIR = SCRIPTS_DIR / "Client"
SERVER_DIR = SCRIPTS_DIR / "Server"
SERVER_SCRIPT = SERVER_DIR / "server_brain.py"


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    description: str
    dataset_file: str
    language_instruction: str


@dataclass(frozen=True)
class ScriptAction:
    name: str
    title: str
    description: str
    script: Path | None


ACTIONS: dict[str, ScriptAction] = {
    "collect_data": ScriptAction(
        name="collect_data",
        title="采集数据",
        description="启动键盘遥操作采集脚本，采集前会先选择任务和采集条数。",
        script=DATA_DIR / "Keyboard_collection.py",
    ),
    "augment_data": ScriptAction(
        name="augment_data",
        title="增强数据",
        description="对已有 HDF5 数据做光照、时间速度、相机扰动等增强。",
        script=DATA_DIR / "Generate_data.py",
    ),
    "replay_data": ScriptAction(
        name="replay_data",
        title="回放数据",
        description="回放 HDF5 轨迹，可按状态或动作模式查看采集结果。",
        script=DATA_DIR / "Replay_collection.py",
    ),
    "inspect_dataset": ScriptAction(
        name="inspect_dataset",
        title="查看数据集",
        description="统计 HDF5 数据集结构、demo 数量、大小和字段。",
        script=DATA_DIR / "inspect_hdf5_dataset.py",
    ),
    "convert_lerobot": ScriptAction(
        name="convert_lerobot",
        title="转换为 LeRobot",
        description="把 IsaacLab HDF5 数据转换为 LeRobot 数据集。",
        script=DATA_DIR / "hdf5_to_lerobot.py",
    ),
    "train_model": ScriptAction(
        name="train_model",
        title="训练模型",
        description="使用 LeRobot 平台训练 ACT 或 SmolVLA 策略模型。",
        script=None,
    ),
    "delete_demo": ScriptAction(
        name="delete_demo",
        title="删除 demo",
        description="从 HDF5 文件中删除指定 demo，默认建议先 dry-run。",
        script=DATA_DIR / "delete_hdf5_demo.py",
    ),
    "register_task": ScriptAction(
        name="register_task",
        title="注册任务",
        description="按固定模板把新的 USD 场景任务写入 task_registry。",
        script=None,
    ),
    "build_environment": ScriptAction(
        name="build_environment",
        title="搭建环境",
        description="打开预设 USD 场景，供用户在 Isaac Sim 中修改并保存，之后可用于注册任务。",
        script=DATA_DIR / "open_usd_environment.py",
    ),
    "analyze_assets": ScriptAction(
        name="analyze_assets",
        title="分析资产",
        description="根据实验室场景描述分析需要的 USD 资产，并列出已有资产和缺失资产。",
        script=None,
    ),
    "run_inference": ScriptAction(
        name="run_inference",
        title="运行推理/评估",
        description="同时启动模型服务和 IsaacLab 推理客户端，运行对应策略评估。",
        script=None,
    ),
    "help": ScriptAction(
        name="help",
        title="帮助",
        description="展示智能体能做什么。",
        script=None,
    ),
}


@lru_cache(maxsize=1)
def _load_task_registry():
    registry_path = DATA_DIR / "task_registry.py"
    if not registry_path.exists():
        raise FileNotFoundError(f"找不到任务注册表: {registry_path}")

    if str(DATA_DIR) not in sys.path:
        sys.path.insert(0, str(DATA_DIR))

    spec = importlib.util.spec_from_file_location("completed_task_registry", registry_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载任务注册表: {registry_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def list_tasks() -> list[TaskInfo]:
    registry = _load_task_registry()
    tasks: list[TaskInfo] = []
    presets = list(getattr(registry, "TASK_PRESETS", {}).values())
    if not presets:
        presets = registry.list_task_presets()
    for preset in presets:
        tasks.append(
            TaskInfo(
                task_id=str(preset.task_id),
                description=str(preset.description),
                dataset_file=str(preset.dataset_file),
                language_instruction=str(preset.language_instruction),
            )
        )
    return tasks


def get_task(task_id: str) -> TaskInfo:
    for task in list_tasks():
        if task.task_id == task_id:
            return task
    known = ", ".join(task.task_id for task in list_tasks())
    raise KeyError(f"未知任务: {task_id}。可选任务: {known}")


def default_task_id() -> str:
    registry = _load_task_registry()
    return str(registry.DEFAULT_TASK_ID)


def reload_task_registry() -> None:
    _load_task_registry.cache_clear()


def action_descriptions() -> list[dict[str, str]]:
    return [
        {
            "name": action.name,
            "title": action.title,
            "description": action.description,
        }
        for action in ACTIONS.values()
    ]


def inference_script_for(policy_type: str) -> Path:
    normalized = policy_type.strip().lower()
    if normalized == "pi0":
        return CLIENT_DIR / "inference_pi0.py"
    if normalized == "smolvla":
        return CLIENT_DIR / "inference_smolvla.py"
    return CLIENT_DIR / "inference_act.py"
