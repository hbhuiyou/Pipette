from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent import (
    PROJECT_ROOT,
    SCRIPTS_DIR,
    generate_task_registration_draft,
    load_local_config,
    python_cmd,
    register_task_in_registry,
    resolve_lerobot_python,
    resolve_lerobot_train_command,
    resolve_lerobot_workdir,
    resolve_project_path,
    sanitize_task_id,
    train_task_slug,
    unique_task_id,
)
from llm_client import OpenAICompatibleClient, fallback_intent
from hunyuan3d_client import (
    Hunyuan3DClient,
    configured_ai3d_output_dir,
    download_result_file,
    list_generated_assets,
    sanitize_asset_name,
    select_result_file,
)
from script_catalog import (
    ACTIONS,
    SERVER_SCRIPT,
    action_descriptions,
    default_task_id,
    get_task,
    inference_script_for,
    list_tasks,
    reload_task_registry,
)
from terminal_launcher import format_command, launch_new_terminal


POLICY_TYPES = ["act", "smolvla", "pi0"]
INFERENCE_TYPES = ["act", "smolvla", "pi0"]
ASSET_REQUIREMENT_RULES = [
    {
        "name": "实验台/工作台",
        "purpose": "作为生物实验室的基础操作台面，用于放置仪器和耗材。",
        "aliases": ["table", "bench", "clean_bench"],
    },
    {
        "name": "生物安全柜/安全柜",
        "purpose": "用于无菌或安全操作区域，适合细胞、样品和试剂处理。",
        "aliases": ["biosafety", "safety_cabinet", "cabinet"],
    },
    {
        "name": "移液器",
        "purpose": "用于液体转移、取样和加样任务。",
        "aliases": ["pipette"],
    },
    {
        "name": "试管/离心管",
        "purpose": "用于保存样品、试剂和离心操作。",
        "aliases": ["tube", "centrifuge_tube"],
    },
    {
        "name": "试管架/离心管架",
        "purpose": "用于固定和摆放试管或离心管。",
        "aliases": ["rack", "tube_rack", "incubation_rack"],
    },
    {
        "name": "培养皿",
        "purpose": "用于样本培养、放置和取出任务。",
        "aliases": ["petri"],
    },
    {
        "name": "离心机",
        "purpose": "用于样品离心、开盖和关盖等实验流程。",
        "aliases": ["centrifuge"],
    },
    {
        "name": "PCR 仪",
        "purpose": "用于 PCR 扩增相关实验流程。",
        "aliases": ["pcr"],
    },
    {
        "name": "培养箱",
        "purpose": "用于细胞或微生物样品恒温培养。",
        "aliases": ["incubator"],
    },
    {
        "name": "电子天平",
        "purpose": "用于称量样品和试剂。",
        "aliases": ["balance"],
    },
    {
        "name": "水浴锅/水浴设备",
        "purpose": "用于样品恒温水浴处理。",
        "aliases": ["water_bath"],
    },
    {
        "name": "酒精灯/火焰灭菌组件",
        "purpose": "用于简单灭菌或模拟实验台基础设备。",
        "aliases": ["alcohol_lamp"],
    },
    {
        "name": "超微量分光光度计",
        "purpose": "用于核酸或蛋白样品浓度检测。",
        "aliases": ["spectrophotometer"],
    },
    {
        "name": "超纯水系统/储水设备",
        "purpose": "用于提供实验用水和水处理场景。",
        "aliases": ["water_production", "water_storage", "ultrapure"],
    },
]


@dataclass
class WebJob:
    id: str
    title: str
    command: str
    pid_file: str
    created_at: float
    terminal_pid: int | None = None
    kind: str = "script"
    progress_path: str = ""
    log_path: str = ""


JOBS: dict[str, WebJob] = {}
LAST_JOB_ID: str | None = None
LAST_JOB_IDS: list[str] = []
HUNYUAN3D_JOBS: dict[str, dict[str, Any]] = {}
HUNYUAN3D_JOBS_LOCK = threading.Lock()


def runtime_dir() -> Path:
    path = Path(__file__).resolve().parent / ".runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    return json.loads(raw or "{}")


def launch_web_job(
    argv: list[str],
    title: str,
    *,
    cwd: Path = PROJECT_ROOT,
    keep_open_default: bool = False,
    clean_python_env: bool = False,
    kind: str = "script",
    progress_path: str = "",
    log_path: str = "",
) -> WebJob:
    global LAST_JOB_ID

    job_id = uuid.uuid4().hex[:12]
    pid_file = runtime_dir() / f"{job_id}.pid"
    env_unset = ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE") if clean_python_env else ()
    process = launch_new_terminal(
        argv,
        cwd=cwd,
        title=title,
        pid_file=pid_file,
        keep_open_default=keep_open_default,
        env_unset=env_unset,
        log_file=Path(log_path) if log_path else None,
    )
    job = WebJob(
        id=job_id,
        title=title,
        command=format_command(argv),
        pid_file=str(pid_file),
        created_at=time.time(),
        terminal_pid=process.pid,
        kind=kind,
        progress_path=progress_path,
        log_path=log_path,
    )
    JOBS[job_id] = job
    LAST_JOB_ID = job_id
    return job


def wait_for_log_text(log_path: Path, needles: tuple[str, ...], timeout_sec: float) -> bool:
    deadline = time.time() + max(0.0, timeout_sec)
    while time.time() <= deadline:
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
            if any(needle.lower() in text for needle in needles):
                return True
        except FileNotFoundError:
            pass
        except Exception:
            pass
        time.sleep(0.5)
    return False


def pid_from_file(path_text: str) -> int | None:
    try:
        text = Path(path_text).read_text(encoding="utf-8").strip()
        return int(text)
    except Exception:
        return None


def stop_pid(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def stop_job(job_id: str | None) -> dict[str, Any]:
    if job_id in {"__all__", "all"}:
        stopped = [stop_job(existing_id) for existing_id in list(JOBS)]
        ok_count = sum(1 for item in stopped if item.get("ok"))
        return {"ok": True, "message": f"已尝试停止 {ok_count} 个 Web Agent 记录的任务。", "stopped": stopped}
    if not job_id and LAST_JOB_IDS:
        stopped = [stop_job(existing_id) for existing_id in list(LAST_JOB_IDS) if existing_id in JOBS]
        ok_count = sum(1 for item in stopped if item.get("ok"))
        if ok_count:
            return {"ok": True, "message": f"已尝试停止最近启动的一组任务，共 {ok_count} 个。", "stopped": stopped}
    if not job_id:
        job_id = LAST_JOB_ID
    if not job_id or job_id not in JOBS:
        return {"ok": False, "message": "当前没有 Web Agent 记录的运行任务。"}
    job = JOBS.pop(job_id)
    pid = pid_from_file(job.pid_file) or job.terminal_pid
    if job.progress_path:
        mark_progress_stopped(job.progress_path)
    if pid is None:
        return {"ok": True, "message": f"任务 {job.title} 没有可停止的 PID，可能已经结束。"}
    stop_pid(pid)
    return {"ok": True, "message": f"已发送停止信号：{job.title}", "job": asdict(job)}


def mark_progress_stopped(progress_path: str) -> None:
    try:
        path = Path(progress_path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {}
        data["status"] = "stopped"
        data["updated_at"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def progress_for_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        return {"ok": False, "message": "找不到这个任务记录。"}
    if not job.progress_path:
        return {"ok": False, "message": "这个任务没有进度文件。"}

    progress_path = Path(job.progress_path)
    if not progress_path.exists():
        return {
            "ok": True,
            "ready": False,
            "job_id": job.id,
            "message": "推理客户端正在启动，等待进度数据。",
        }

    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "message": f"读取进度失败: {exc}"}

    return {"ok": True, "ready": True, "job_id": job.id, "progress": data}


def augment_stats_for_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        return {"ok": False, "message": "找不到这个任务记录。"}
    if not job.log_path:
        return {"ok": False, "message": "这个增强任务没有统计日志。"}

    log_path = Path(job.log_path)
    if not log_path.exists():
        return {"ok": True, "job_id": job.id, "stats": {"success": 0, "failure": 0, "total": 0, "done": False}}

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {"ok": False, "message": f"读取增强统计失败: {exc}"}

    stats = parse_augment_log_stats(text)
    return {"ok": True, "job_id": job.id, "stats": stats}


def inference_stats_for_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        return {"ok": False, "message": "找不到这个任务记录。"}
    if not job.log_path:
        return {"ok": False, "message": "这个推理任务没有统计日志。"}

    log_path = Path(job.log_path)
    if not log_path.exists():
        return {"ok": True, "job_id": job.id, "stats": {"success": 0, "failure": 0, "total": 0, "done": False}}

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {"ok": False, "message": f"读取推理统计失败: {exc}"}

    return {"ok": True, "job_id": job.id, "stats": parse_inference_log_stats(text)}


def convert_progress_for_job(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)
    if job is None:
        return {"ok": False, "message": "找不到这个任务记录。"}
    if not job.log_path:
        return {"ok": False, "message": "这个转换任务没有进度日志。"}

    log_path = Path(job.log_path)
    if not log_path.exists():
        return {"ok": True, "job_id": job.id, "progress": {"completed": 0, "total": 0, "kept_frames": 0, "done": False}}

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        return {"ok": False, "message": f"读取转换进度失败: {exc}"}

    return {"ok": True, "job_id": job.id, "progress": parse_convert_log_progress(text)}


def parse_augment_log_stats(text: str) -> dict[str, int | bool]:
    success = len(re.findall(r"\[INFO\]\s+Wrote\s+demo_\d+", text))
    failure = len(re.findall(r"\[INFO\]\s+Dropped variant\b", text))

    worker_totals = [
        int(value)
        for value in re.findall(r"\[INFO\]\s+(?:Launching|Running) worker \d+/(\d+)\b", text)
    ]
    worker_total = max(worker_totals, default=0)
    completed_workers = len(re.findall(r"\[INFO\]\s+Augmentation completed\.", text))

    done = (
        (worker_total > 0 and completed_workers >= worker_total)
        or "One or more workers failed:" in text
        or "Worker failed (code=" in text
        or "Interrupted by user" in text
    )
    return {
        "success": success,
        "failure": failure,
        "total": success + failure,
        "workers": worker_total,
        "completed_workers": completed_workers,
        "done": done,
    }


def parse_inference_log_stats(text: str) -> dict[str, int | bool]:
    outcomes = re.findall(r"(?m)^\[EP\s+\d+\]\s+([a-z_]+)\s+\(", text)
    success = sum(1 for outcome in outcomes if outcome == "success")
    failure = len(outcomes) - success
    done = "Evaluation Summary ==========" in text or "Traceback (most recent call last)" in text
    return {"success": success, "failure": failure, "total": len(outcomes), "done": done}


def parse_convert_log_progress(text: str) -> dict[str, int | bool]:
    completed = 0
    total = 0
    kept_frames = 0

    total_match = re.search(r"\[INFO\]\s+Total episodes:\s*(\d+)", text)
    if total_match:
        total = int(total_match.group(1))

    for match in re.finditer(r"\[PROGRESS\]\s+Converted episodes:\s*(\d+)/(\d+),\s*kept_frames=(\d+)", text):
        completed = int(match.group(1))
        total = int(match.group(2))
        kept_frames = int(match.group(3))

    final_frames = re.search(r"\[INFO\]\s+Total kept frames:\s*(\d+)", text)
    if final_frames:
        kept_frames = int(final_frames.group(1))

    done = "[SUCCESS] LeRobot Dataset created successfully" in text or "[ERROR]" in text or "Traceback" in text
    return {"completed": completed, "total": total, "kept_frames": kept_frames, "done": done}


def task_options() -> list[dict[str, str]]:
    return [asdict(task) for task in list_tasks()]


def field(name: str, label: str, kind: str, default: Any = "", **extra: Any) -> dict[str, Any]:
    data = {"name": name, "label": label, "type": kind, "default": default}
    data.update(extra)
    return data


def train_defaults(policy_type: str) -> tuple[int, int]:
    if policy_type == "act":
        return 32, 15000
    if policy_type == "smolvla":
        return 8, 13000
    if policy_type == "pi0":
        return 4, 20000
    return 8, 20000


def configured_asset_dir() -> str:
    return str(resolve_project_path(os.getenv("AGENT_ASSET_DIR", "Asset")))


def ai3d_output_dir() -> Path:
    return configured_ai3d_output_dir(PROJECT_ROOT)


def submit_hunyuan3d_job(params: dict[str, Any]) -> dict[str, Any]:
    prompt = str(params.get("prompt") or "").strip()
    image_base64 = str(params.get("image_base64") or "").strip()
    multi_view_params = params.get("multi_view_images") or []
    if not isinstance(multi_view_params, list):
        raise ValueError("多视图图片参数格式无效。")
    multi_view_images: list[dict[str, str]] = []
    multi_view_names: list[dict[str, str]] = []
    for item in multi_view_params:
        if not isinstance(item, dict):
            raise ValueError("多视图图片参数格式无效。")
        view_type = str(item.get("view_type") or "").strip().lower()
        image_content = str(item.get("image_base64") or "").strip()
        image_name = str(item.get("image_name") or "").strip()
        multi_view_images.append(
            {
                "ViewType": view_type,
                "ViewImageBase64": image_content,
            }
        )
        multi_view_names.append(
            {
                "view_type": view_type,
                "image_name": image_name,
            }
        )
    asset_name = sanitize_asset_name(str(params.get("asset_name") or ""), "USDZ")
    model = str(params.get("model") or "3.0").strip() or "3.0"
    generate_type = str(params.get("generate_type") or "Normal").strip() or "Normal"
    face_count_value = params.get("face_count")
    face_count = None
    if face_count_value not in {None, ""}:
        face_count = int(face_count_value)

    client = Hunyuan3DClient()
    response = client.submit_pro_job(
        prompt=prompt,
        image_base64=image_base64,
        multi_view_images=multi_view_images,
        model=model,
        generate_type=generate_type,
        result_format="USDZ",
        face_count=face_count,
        enable_pbr=bool_param(params, "enable_pbr", False),
    )
    cloud_job_id = str(response.get("JobId") or "").strip()
    if not cloud_job_id:
        raise RuntimeError("腾讯云已接收请求，但没有返回 JobId。")

    return register_hunyuan3d_job(
        cloud_job_id,
        prompt=prompt,
        image_name=str(params.get("image_name") or "").strip(),
        multi_view_names=multi_view_names,
        asset_name=asset_name,
        generate_type=generate_type,
        model=model,
    )


def register_hunyuan3d_job(
    cloud_job_id: str,
    *,
    prompt: str = "",
    image_name: str = "",
    multi_view_names: list[dict[str, str]] | None = None,
    asset_name: str = "",
    generate_type: str = "",
    model: str = "",
) -> dict[str, Any]:
    cloud_job_id = str(cloud_job_id or "").strip()
    if not cloud_job_id:
        raise ValueError("腾讯混元生3D任务号不能为空。")
    generation_id = uuid.uuid4().hex[:12]
    now = time.time()
    job = {
        "id": generation_id,
        "cloud_job_id": cloud_job_id,
        "status": "submitted",
        "cloud_state": "WAIT",
        "progress": "",
        "progress_available": False,
        "prompt": prompt,
        "image_name": image_name,
        "multi_view_names": list(multi_view_names or []),
        "asset_name": asset_name,
        "generate_type": generate_type,
        "model": model,
        "result_format": "USDZ",
        "output_dir": str(ai3d_output_dir()),
        "local_path": "",
        "preview_url": "",
        "error": "",
        "created_at": now,
        "updated_at": now,
    }
    with HUNYUAN3D_JOBS_LOCK:
        HUNYUAN3D_JOBS[generation_id] = job
    return dict(job)


def recover_hunyuan3d_job(params: dict[str, Any]) -> dict[str, Any]:
    return register_hunyuan3d_job(str(params.get("cloud_job_id") or ""))


def hunyuan3d_job_status(generation_id: str) -> dict[str, Any]:
    with HUNYUAN3D_JOBS_LOCK:
        existing = HUNYUAN3D_JOBS.get(generation_id)
        job = dict(existing) if existing else None
    if job is None:
        raise ValueError("找不到这个混元生3D生成任务。")
    if job["status"] in {"completed", "failed"}:
        return job

    try:
        response = Hunyuan3DClient().query_pro_job(str(job["cloud_job_id"]))
        cloud_state = str(response.get("Status") or "WAIT").upper()
        job["cloud_state"] = cloud_state
        job["progress_available"] = False
        job["updated_at"] = time.time()
        job["credit_consumed"] = response.get("ResultCreditConsumed")
        job["credit_details"] = str(response.get("ResultCreditDetails") or "")

        if cloud_state == "DONE":
            result_file = select_result_file(list(response.get("ResultFile3Ds") or []), "USDZ")
            output_path = download_result_file(
                result_file,
                ai3d_output_dir(),
                job_id=str(job["cloud_job_id"]),
                result_format="USDZ",
                asset_name=str(job.get("asset_name") or ""),
            )
            job["status"] = "completed"
            job["progress"] = "100"
            job["local_path"] = str(output_path)
            job["preview_url"] = str(result_file.get("PreviewImageUrl") or "")
            job["model_url"] = str(result_file.get("Url") or "")
            job["error"] = ""
        elif cloud_state == "FAIL":
            job["status"] = "failed"
            job["error"] = str(response.get("ErrorMessage") or response.get("ErrorCode") or "模型生成失败。")
        elif cloud_state == "WAIT":
            job["status"] = "queued"
            job["progress"] = ""
            job["error"] = ""
        else:
            job["status"] = "running"
            job["progress"] = ""
            job["error"] = ""
    except Exception as exc:
        job["status"] = "download_error" if job.get("cloud_state") == "DONE" else "query_error"
        job["error"] = str(exc)
        job["updated_at"] = time.time()

    with HUNYUAN3D_JOBS_LOCK:
        HUNYUAN3D_JOBS[generation_id] = job
    return dict(job)


def hunyuan3d_assets() -> dict[str, Any]:
    output_dir = ai3d_output_dir()
    return {
        "ok": True,
        "output_dir": str(output_dir),
        "assets": list_generated_assets(output_dir),
    }


def create_hunyuan3d_physics_asset(params: dict[str, Any]) -> dict[str, Any]:
    output_dir = ai3d_output_dir().resolve()
    usdz_path = Path(str(params.get("usdz_path") or "")).expanduser().resolve()
    try:
        usdz_path.relative_to(output_dir)
    except ValueError as exc:
        raise ValueError("只能处理 AI3D 输出目录中的 USDZ 模型。") from exc
    if usdz_path.suffix.lower() != ".usdz" or not usdz_path.is_file():
        raise ValueError("请选择一个存在的 USDZ 模型。")

    tool_path = SCRIPTS_DIR / "Asset" / "create_physics_asset.py"
    if not tool_path.is_file():
        raise RuntimeError(f"找不到 USD 物理资产生成工具: {tool_path}")
    command = python_cmd(
        tool_path,
        [
            str(usdz_path),
            "--mass",
            str(float(params.get("mass") or 0)),
            "--collision-type",
            str(params.get("collision_type") or ""),
            "--friction",
            str(float(params.get("friction") or 0)),
            "--body-type",
            str(params.get("body_type") or ""),
        ],
    )
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(
            f"无法启动物理 USD Python: {command[0]}。"
            f"工作目录: {PROJECT_ROOT}。错误: {exc}"
        ) from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "未知错误").strip()
        raise RuntimeError(f"物理 USD 生成失败: {details[-1000:]}")
    result_prefix = "PHYSICS_USD_RESULT="
    result_line = next(
        (
            line
            for line in reversed((completed.stdout or "").splitlines())
            if line.startswith(result_prefix)
        ),
        "",
    )
    try:
        result = json.loads(result_line[len(result_prefix) :])
    except json.JSONDecodeError as exc:
        details = (completed.stderr or completed.stdout or "no output").strip()
        raise RuntimeError(f"物理 USD 工具未返回有效结果: {details[-1000:]}") from exc
    return {
        "ok": True,
        "result": result,
        "assets": list_generated_assets(output_dir),
    }


def scan_usd_assets(asset_dir: str) -> list[str]:
    root = Path(asset_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []
    return sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )


def _asset_matches(aliases: list[str], asset_name: str) -> bool:
    normalized = asset_name.lower().replace("-", "_").replace(" ", "_")
    return any(alias.lower() in normalized for alias in aliases)


def local_asset_analysis(lab_description: str, owned_assets: list[str]) -> dict[str, Any]:
    needed_assets: list[dict[str, Any]] = []
    owned_matches: list[dict[str, str]] = []
    missing_assets: list[dict[str, str]] = []

    matched_files_seen: set[str] = set()
    for rule in ASSET_REQUIREMENT_RULES:
        matches = [asset for asset in owned_assets if _asset_matches(list(rule["aliases"]), asset)]
        needed_assets.append({
            "name": rule["name"],
            "purpose": rule["purpose"],
            "matched_files": matches,
        })
        if matches:
            for file_name in matches:
                if file_name not in matched_files_seen:
                    matched_files_seen.add(file_name)
                    owned_matches.append({"file": file_name, "use": str(rule["purpose"])})
        else:
            missing_assets.append({"name": str(rule["name"]), "reason": "当前 asset 文件夹中未找到明显匹配的 USD 文件。"})

    return {
        "source": "local",
        "summary": f"已根据描述“{lab_description}”和本地 USD 文件名生成资产规划。",
        "needed_assets": needed_assets,
        "owned_assets": owned_matches,
        "missing_assets": missing_assets,
        "suggestions": [
            "已有资产可直接拖入或引用到实验室 USD 场景中。",
            "缺失资产建议先补齐 USD 文件，再进入注册任务流程。",
        ],
    }


def analyze_lab_asset_plan(lab_description: str, asset_dir: str) -> dict[str, Any]:
    owned_assets = scan_usd_assets(asset_dir)
    local_plan = local_asset_analysis(lab_description, owned_assets)
    llm_plan = OpenAICompatibleClient().analyze_lab_assets(lab_description, owned_assets)
    if llm_plan.get("source") == "api":
        plan = llm_plan
    else:
        plan = local_plan
        if llm_plan.get("summary"):
            plan["api_warning"] = llm_plan["summary"]
    plan["asset_dir"] = asset_dir
    plan["all_assets"] = owned_assets
    return plan


def extract_count_from_text(text: str) -> int | None:
    patterns = (
        r"(\d+)\s*(?:条|个|次)?\s*(?:demo|demos|演示|轨迹|数据)",
        r"(?:采集|收集|record|collect)\D{0,12}(\d+)",
        r"(\d+)\D{0,12}(?:采集|收集|record|collect)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if match:
            try:
                value = int(match.group(1))
            except Exception:
                continue
            if value > 0:
                return value
    return None


def form_for_action(action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = params or {}
    tasks = task_options()
    task_default = str(params.get("task_id") or default_task_id())
    task_select = field("task_id", "任务", "select", task_default, options=[
        {"value": t["task_id"], "label": f"{t['task_id']}  [{t['dataset_file']}]"} for t in tasks
    ])

    if action == "collect_data":
        return {
            "title": "采集数据",
            "description": "选择任务和采集条数，启动 IsaacLab 键盘遥操作采集终端。",
            "fields": [
                task_select,
                field("dataset_file", "数据保存路径", "text", str(params.get("dataset_file") or get_task(task_default).dataset_file)),
                field("num_demos", "采集 demo 条数", "number", int(params.get("num_demos") or 5), min=1),
                field("gui", "有界面运行", "checkbox", True),
            ],
        }
    if action == "build_environment":
        default_usd = str(resolve_project_path(os.getenv("AGENT_ENV_TEMPLATE_USD", "Asset/lab.usd")))
        default_asset_dir = configured_asset_dir()
        return {
            "title": "搭建环境",
            "description": "打开预设 USD 场景，在 Isaac Sim 中修改并另存，之后用于注册任务。",
            "fields": [
                field("usd_path", "USD 场景路径", "text", str(params.get("usd_path") or default_usd)),
                field("asset_dir", "资源文件夹", "text", str(params.get("asset_dir") or default_asset_dir)),
            ],
        }
    if action == "analyze_assets":
        return {
            "title": "分析资产",
            "description": "输入要搭建的实验室或任务场景描述，系统会扫描 asset 文件夹并分析已有资产和缺失资产。",
            "fields": [
                field("lab_description", "实验室/场景描述", "textarea", str(params.get("lab_description") or params.get("description") or "")),
                field("asset_dir", "USD 资产文件夹", "text", str(params.get("asset_dir") or configured_asset_dir())),
            ],
        }
    if action == "register_task":
        return {
            "title": "注册任务",
            "description": "输入保存后的 USD 路径和任务描述，系统生成 task_id 并写入 task_registry。",
            "fields": [
                field("usd_path", "搭建好的 USD 场景路径", "text", str(params.get("usd_path") or "")),
                field("description", "任务描述", "textarea", str(params.get("description") or "")),
            ],
        }
    if action == "convert_lerobot":
        return {
            "title": "转换 LeRobot",
            "description": "选择任务和 HDF5 路径，其余参数使用推荐默认值。",
            "fields": [
                field("task_optional", "使用任务预设", "checkbox", True),
                task_select,
                field("hdf5_path", "HDF5 输入路径", "text", ""),
                field("output_dir", "LeRobot 输出目录", "text", ""),
            ],
        }
    if action == "train_model":
        task_slug = train_task_slug(task_default)
        model_type = str(params.get("policy_type") or "act")
        model_root = resolve_project_path(os.getenv("LEROBOT_MODEL_ROOT", "models"))
        return {
            "title": "训练模型",
            "description": "使用 LeRobot 平台训练 ACT、SmolVLA 或 PI0。",
            "fields": [
                task_select,
                field("policy_type", "模型类型", "select", model_type, options=[
                    {"value": "act", "label": "ACT"},
                    {"value": "smolvla", "label": "SmolVLA"},
                    {"value": "pi0", "label": "PI0"},
                ]),
                field("dataset_repo_id", "LeRobot 数据集目录（绝对路径）", "text", str(params.get("dataset_repo_id") or "")),
                field("output_dir", "模型输出目录", "text", str(Path(model_root) / f"{task_slug}_{model_type}")),
                field("batch_size", "batch size", "number", train_defaults(model_type)[0], min=1),
                field("steps", "训练步数 steps", "number", train_defaults(model_type)[1], min=1),
                field("pi0_freeze_vision_encoder", "PI0: freeze vision encoder", "checkbox", True, show_if="policy_type", show_value="pi0"),
                field("pi0_train_expert_only", "PI0: train expert only", "checkbox", True, show_if="policy_type", show_value="pi0"),
                field("pi0_gradient_checkpointing", "PI0: gradient checkpointing", "checkbox", True, show_if="policy_type", show_value="pi0"),
                field("wandb_enable", "启用 wandb", "checkbox", False),
                field("wandb_project", "wandb project", "text", "lerobot", show_if="wandb_enable"),
                field("wandb_entity", "wandb entity / 账号", "text", "", show_if="wandb_enable"),
                field("wandb_notes", "wandb notes", "textarea", "", show_if="wandb_enable"),
                field("push_to_hub", "上传到 Hugging Face Hub", "checkbox", False),
                field(
                    "hub_repo_id",
                    "Hugging Face repo id（账号/模型名）",
                    "text",
                    f"your-hf-username/{model_type}_{task_slug}",
                    show_if="push_to_hub",
                ),
                field("hub_private", "Hub 私有仓库", "checkbox", False, show_if="push_to_hub"),
            ],
        }
    if action == "run_inference":
        return {
            "title": "推理/评估",
            "description": "先自动启动模型服务，再启动 IsaacSim 推理客户端。",
            "fields": [
                task_select,
                field("policy_type", "模型/客户端类型", "select", str(params.get("policy_type") or "act"), options=[
                    {"value": "act", "label": "ACT"},
                    {"value": "smolvla", "label": "SmolVLA"},
                    {"value": "pi0", "label": "PI0"},
                ]),
                field("policy_path", "模型路径或 Hugging Face repo id", "text", str(params.get("policy_path") or "")),
                field("episodes", "评估回合数", "number", int(params.get("episodes") or 100), min=1),
                field("gui", "有界面运行", "checkbox", True),
            ],
        }
    if action == "augment_data":
        task = get_task(task_default)
        return {
            "title": "增强数据",
            "description": "对已有 HDF5 数据做光照、时间速度、相机扰动等增强。",
            "fields": [
                task_select,
                field("dataset_file", "HDF5 数据路径", "text", task.dataset_file),
                field("output_file", "输出文件路径（可空）", "text", ""),
                field("worker_count", "并行 worker 数量", "number", 3, min=1, max=4),
                field("light_scales", "光照增强倍率", "text", "0.8"),
                field("temporal_scales", "时间速度倍率", "text", "1.2"),
                field("camera_jitter_count", "相机扰动数量", "number", 0, min=0),
                field("include_original", "包含原始 demo", "checkbox", True),
                field("gui", "有界面运行", "checkbox", False),
            ],
        }
    if action == "replay_data":
        task = get_task(task_default)
        return {
            "title": "回放数据",
            "description": "回放 HDF5 轨迹，检查采集结果。",
            "fields": [
                task_select,
                field("dataset_file", "HDF5 数据路径", "text", task.dataset_file),
                field("demo_index", "demo 编号，-1 表示全部", "number", -1),
                field("replay_mode", "回放模式", "select", "state", options=[
                    {"value": "state", "label": "state"},
                    {"value": "action", "label": "action"},
                ]),
                field("speed", "回放速度倍率", "text", "1.0"),
            ],
        }
    if action == "inspect_dataset":
        task = get_task(task_default)
        return {
            "title": "查看数据集",
            "description": "统计 HDF5 数据集结构、demo 数量、大小和字段。",
            "fields": [
                field("dataset_file", "HDF5 数据路径", "text", task.dataset_file),
                field("show_tree", "打印完整 HDF5 树", "checkbox", False),
                field("show_attrs", "打印文件属性", "checkbox", True),
            ],
        }
    if action == "delete_demo":
        task = get_task(task_default)
        return {
            "title": "删除 demo",
            "description": "从 HDF5 文件中删除指定 demo，默认先 dry-run。",
            "fields": [
                field("dataset_file", "HDF5 文件", "text", task.dataset_file),
                field("indices", "demo 编号，逗号分隔", "text", ""),
                field("dry_run", "仅预览，不实际删除", "checkbox", True),
            ],
        }
    return {"title": "帮助", "description": "请选择一个动作。", "fields": []}


def classify_text(text: str) -> dict[str, Any]:
    llm = OpenAICompatibleClient()
    local_intent = fallback_intent(text)
    if local_intent.action == "exit":
        intent = local_intent
    elif llm.available:
        intent = llm.classify_intent(text, action_descriptions(), task_options())
        if intent.source != "api" or intent.confidence < 0.45:
            if local_intent.confidence >= intent.confidence:
                local_intent.reply = intent.reply or local_intent.reply
                intent = local_intent
    else:
        intent = local_intent

    action = intent.action
    parameters = dict(intent.parameters or {})
    if action == "collect_data":
        count = extract_count_from_text(text)
        if count is not None:
            parameters["num_demos"] = count
    if action == "analyze_assets" and not parameters.get("lab_description"):
        parameters["lab_description"] = text
    return {
        "action": action,
        "confidence": intent.confidence,
        "parameters": parameters,
        "reply": intent.reply,
        "source": intent.source,
    }


def bool_param(params: dict[str, Any], name: str, default: bool = False) -> bool:
    value = params.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def int_param(params: dict[str, Any], name: str, default: int) -> int:
    try:
        return int(params.get(name, default))
    except Exception:
        return int(default)


def dataset_file_param(params: dict[str, Any], task) -> str:
    dataset_file = str(params.get("dataset_file") or "").strip()
    if not dataset_file:
        return task.dataset_file

    other_default_files = {
        item.dataset_file
        for item in list_tasks()
        if item.task_id != task.task_id
    }
    if dataset_file in other_default_files:
        return task.dataset_file

    return dataset_file


def require_absolute_path_param(params: dict[str, Any], name: str, label: str) -> str:
    value = str(params.get(name) or "").strip()
    if not value:
        raise ValueError(f"需要填写{label}，并使用绝对路径。")
    path = Path(value).expanduser()
    if value.startswith("/"):
        return value
    if not path.is_absolute():
        raise ValueError(f"{label}必须是绝对路径：{value}")
    return str(path)


def launch_action(action: str, params: dict[str, Any]) -> dict[str, Any]:
    global LAST_JOB_IDS
    jobs: list[dict[str, Any]] = []
    message = ""

    if action == "collect_data":
        task = get_task(str(params.get("task_id") or default_task_id()))
        num_demos = max(1, int_param(params, "num_demos", 5))
        gui = bool_param(params, "gui", True)
        dataset_file = str(params.get("dataset_file") or task.dataset_file).strip() or task.dataset_file
        args = ["--task_id", task.task_id, "--dataset_file", dataset_file, "--num_demos", str(num_demos)]
        if not gui:
            args.append("--headless")
        job = launch_web_job(python_cmd(ACTIONS[action].script, args), "采集数据")
        jobs.append(asdict(job))
        message = f"已启动采集终端：{task.task_id}，目标 {num_demos} 条 demo。"

    elif action == "build_environment":
        default_usd = resolve_project_path(os.getenv("AGENT_ENV_TEMPLATE_USD", "Asset/lab.usd"))
        default_asset_dir = resolve_project_path(os.getenv("AGENT_ASSET_DIR", "Asset"))
        usd_path = str(resolve_project_path(str(params.get("usd_path") or default_usd)))
        asset_dir = str(resolve_project_path(str(params.get("asset_dir") or default_asset_dir)))
        job = launch_web_job(
            python_cmd(ACTIONS[action].script, ["--usd-path", usd_path, "--asset-dir", asset_dir]),
            "搭建环境",
            keep_open_default=True,
        )
        jobs.append(asdict(job))
        message = "已打开搭建环境终端。请在 Isaac Sim 中修改并另存 USD。"

    elif action == "analyze_assets":
        lab_description = str(params.get("lab_description") or "").strip()
        asset_dir = str(params.get("asset_dir") or configured_asset_dir()).strip()
        if not lab_description:
            raise ValueError("分析资产需要填写实验室或场景描述。")
        plan = analyze_lab_asset_plan(lab_description, asset_dir)
        LAST_JOB_IDS = []
        return {
            "ok": True,
            "message": "已完成资产分析。",
            "asset_plan": plan,
            "jobs": [],
        }

    elif action == "register_task":
        usd_path = str(params.get("usd_path") or "").strip()
        description = str(params.get("description") or "").strip()
        if not usd_path or not description:
            raise ValueError("注册任务需要 USD 场景路径和任务描述。")
        if not usd_path.lower().endswith((".usd", ".usda", ".usdc")):
            raise ValueError("USD 场景路径应以 .usd、.usda 或 .usdc 结尾。")
        draft = generate_task_registration_draft(description)
        task_id = unique_task_id(sanitize_task_id(draft.get("task_id") or "custom_task"))
        language_instruction = draft.get("language_instruction") or task_id
        normalized_description = draft.get("description") or description
        dataset_file = f"./datasets/{task_id}.hdf5"
        register_task_in_registry(
            task_id=task_id,
            usd_path=usd_path,
            description=normalized_description,
            language_instruction=language_instruction,
            dataset_file=dataset_file,
        )
        reload_task_registry()
        LAST_JOB_IDS = []
        message = f"已注册任务：{task_id}"
        return {
            "ok": True,
            "message": message,
            "registered": {
                "task_id": task_id,
                "usd_path": usd_path,
                "description": normalized_description,
                "language_instruction": language_instruction,
                "dataset_file": dataset_file,
            },
            "jobs": [],
        }

    elif action == "convert_lerobot":
        use_task = bool_param(params, "task_optional", True)
        task_id = str(params.get("task_id") or default_task_id())
        task = get_task(task_id) if use_task else None
        hdf5_path = str(params.get("hdf5_path") or (task.dataset_file if task else ""))
        if not hdf5_path:
            raise ValueError("需要 HDF5 输入路径。")
        task_name = task.task_id if task else Path(hdf5_path).stem
        output_dir = require_absolute_path_param(params, "output_dir", "LeRobot 输出目录")
        args = [
            "--hdf5-path", str(resolve_project_path(hdf5_path)),
            "--output-dir", output_dir,
            "--repo-id", task_name,
            "--fps", "10",
            "--frame-filter", "fresh",
            "--stride", "3",
        ]
        convert_log_path = runtime_dir() / f"convert_{uuid.uuid4().hex[:12]}.log"
        job = launch_web_job(
            python_cmd(ACTIONS[action].script, args, python_exe=resolve_lerobot_python()),
            "转换 LeRobot",
            cwd=resolve_lerobot_workdir(),
            keep_open_default=True,
            clean_python_env=True,
            log_path=str(convert_log_path),
        )
        jobs.append(asdict(job))
        message = f"已启动 LeRobot 转换：{task_name}"

    elif action == "train_model":
        task = get_task(str(params.get("task_id") or default_task_id()))
        policy_type = str(params.get("policy_type") or "act").lower()
        if policy_type not in POLICY_TYPES:
            raise ValueError(f"训练模型类型只能是: {', '.join(POLICY_TYPES)}")
        task_slug = train_task_slug(task.task_id)
        dataset_repo_id = require_absolute_path_param(params, "dataset_repo_id", "LeRobot 数据集目录")
        model_root = resolve_project_path(os.getenv("LEROBOT_MODEL_ROOT", "models"))
        output_dir = str(params.get("output_dir") or model_root / f"{task_slug}_{policy_type}")
        default_batch_size, default_steps = train_defaults(policy_type)
        batch_size = max(1, int_param(params, "batch_size", default_batch_size))
        steps = max(1, int_param(params, "steps", default_steps))
        wandb_enable = bool_param(params, "wandb_enable", False)
        push_to_hub = bool_param(params, "push_to_hub", False)
        policy_repo_id = f"local/{policy_type}_{task_slug}"
        if push_to_hub:
            policy_repo_id = str(params.get("hub_repo_id") or "").strip()
            if not policy_repo_id or "/" not in policy_repo_id:
                raise ValueError("开启 push_to_hub 后，需要填写 Hugging Face repo id，格式为：账号/模型名。")
        argv = [
            resolve_lerobot_train_command(),
            f"--dataset.repo_id={dataset_repo_id}",
            f"--policy.type={policy_type}",
            f"--policy.repo_id={policy_repo_id}",
            f"--output_dir={output_dir}",
            f"--batch_size={batch_size}",
            f"--steps={steps}",
            f"--wandb.enable={str(wandb_enable).lower()}",
            f"--policy.push_to_hub={str(push_to_hub).lower()}",
        ]
        if wandb_enable:
            wandb_project = str(params.get("wandb_project") or "lerobot").strip()
            wandb_entity = str(params.get("wandb_entity") or "").strip()
            wandb_notes = str(params.get("wandb_notes") or "").strip()
            if wandb_project:
                argv.append(f"--wandb.project={wandb_project}")
            if wandb_entity:
                argv.append(f"--wandb.entity={wandb_entity}")
            if wandb_notes:
                argv.append(f"--wandb.notes={wandb_notes}")
        if push_to_hub:
            argv.append(f"--policy.private={str(bool_param(params, 'hub_private', False)).lower()}")
        if policy_type == "pi0":
            argv.append(f"--policy.freeze_vision_encoder={str(bool_param(params, 'pi0_freeze_vision_encoder', True)).lower()}")
            argv.append(f"--policy.train_expert_only={str(bool_param(params, 'pi0_train_expert_only', True)).lower()}")
            argv.append(f"--policy.gradient_checkpointing={str(bool_param(params, 'pi0_gradient_checkpointing', True)).lower()}")
        job = launch_web_job(
            argv,
            "训练模型",
            cwd=resolve_lerobot_workdir(),
            keep_open_default=True,
            clean_python_env=True,
        )
        jobs.append(asdict(job))
        message = f"已启动训练：{policy_type} / {task.task_id}"

    elif action == "run_inference":
        task = get_task(str(params.get("task_id") or default_task_id()))
        policy_type = str(params.get("policy_type") or "act").lower()
        if policy_type not in INFERENCE_TYPES:
            raise ValueError(f"推理模型类型只能是: {', '.join(INFERENCE_TYPES)}")
        policy_path = str(params.get("policy_path") or "").strip()
        if not policy_path:
            raise ValueError("推理需要模型路径或 Hugging Face repo id。")
        episodes = max(1, int_param(params, "episodes", 100))
        gui = bool_param(params, "gui", True)
        endpoint = "tcp://127.0.0.1:5555"
        server_args = ["--policy-path", policy_path, "--policy-type", policy_type, "--bind", endpoint, "--device", "cuda"]
        server_log_path = runtime_dir() / f"server_{uuid.uuid4().hex[:12]}.log"
        server_job = launch_web_job(
            python_cmd(SERVER_SCRIPT, server_args, python_exe=resolve_lerobot_python()),
            "模型服务",
            cwd=resolve_lerobot_workdir(),
            keep_open_default=True,
            clean_python_env=True,
            log_path=str(server_log_path),
        )
        jobs.append(asdict(server_job))
        server_ready_timeout = float(os.getenv("INFERENCE_SERVER_READY_TIMEOUT_SEC", "300"))
        if not wait_for_log_text(server_log_path, ("Inference device: cuda",), server_ready_timeout):
            LAST_JOB_IDS = [server_job.id]
            return {
                "ok": True,
                "message": f"模型服务已启动，但 {int(server_ready_timeout)} 秒内未检测到服务端打印 cuda，暂未启动推理客户端。请查看模型服务终端输出。",
                "jobs": jobs,
            }
        client_args = [
            "--task-id",
            task.task_id,
            "--server-endpoint",
            endpoint,
            "--episodes",
            str(episodes),
        ]
        if not gui:
            client_args.append("--headless")
        inference_log_path = runtime_dir() / f"inference_{uuid.uuid4().hex[:12]}.log"
        client_job = launch_web_job(
            python_cmd(inference_script_for(policy_type), client_args),
            "运行推理",
            kind="inference",
            log_path=str(inference_log_path),
        )
        jobs.append(asdict(client_job))
        message = f"已启动模型服务和推理客户端：{policy_type} / {task.task_id}"

    elif action == "augment_data":
        task = get_task(str(params.get("task_id") or default_task_id()))
        dataset_file = dataset_file_param(params, task)
        args = [
            "--task_id", task.task_id,
            "--dataset_file", dataset_file,
            "--num_envs", str(max(1, int_param(params, "worker_count", 3))),
            "--light_intensity_scales", str(params.get("light_scales") or "0.8"),
            "--temporal_speed_scales", str(params.get("temporal_scales") or "1.2"),
            "--camera_jitter_count", str(max(0, int_param(params, "camera_jitter_count", 0))),
        ]
        if params.get("output_file"):
            args += ["--output_file", str(params["output_file"])]
        if bool_param(params, "include_original", True):
            args.append("--include_original")
        if "gui" in params:
            headless = not bool_param(params, "gui", False)
        else:
            headless = bool_param(params, "headless", True)
        if headless:
            args.append("--headless")
        augment_log_path = runtime_dir() / f"augment_{uuid.uuid4().hex[:12]}.log"
        job = launch_web_job(
            python_cmd(ACTIONS[action].script, args),
            "增强数据",
            log_path=str(augment_log_path),
        )
        jobs.append(asdict(job))
        message = f"已启动数据增强：{task.task_id}"

    elif action == "replay_data":
        task = get_task(str(params.get("task_id") or default_task_id()))
        dataset_file = dataset_file_param(params, task)
        args = [
            "--task_id", task.task_id,
            "--dataset_file", dataset_file,
            "--demo_index", str(int_param(params, "demo_index", -1)),
            "--replay_mode", str(params.get("replay_mode") or "state"),
            "--speed", str(params.get("speed") or "1.0"),
        ]
        job = launch_web_job(python_cmd(ACTIONS[action].script, args), "回放数据")
        jobs.append(asdict(job))
        message = f"已启动回放：{task.task_id}"

    elif action == "inspect_dataset":
        dataset_file = str(params.get("dataset_file") or "")
        if not dataset_file:
            raise ValueError("需要 HDF5 数据路径。")
        args = ["--file", dataset_file]
        if bool_param(params, "show_tree", False):
            args.append("--show-tree")
        if bool_param(params, "show_attrs", True):
            args.append("--show-attrs")
        job = launch_web_job(python_cmd(ACTIONS[action].script, args), "查看数据集")
        jobs.append(asdict(job))
        message = "已启动数据集查看终端。"

    elif action == "delete_demo":
        dataset_file = str(params.get("dataset_file") or "")
        indices = str(params.get("indices") or "")
        if not dataset_file or not indices:
            raise ValueError("删除 demo 需要 HDF5 文件和 demo 编号。")
        args = ["--file", dataset_file, "--indices", indices]
        if bool_param(params, "dry_run", True):
            args.append("--dry-run")
        job = launch_web_job(python_cmd(ACTIONS[action].script, args), "删除 demo")
        jobs.append(asdict(job))
        message = "已启动删除 demo 终端。"

    else:
        raise ValueError(f"暂不支持这个动作：{action}")

    LAST_JOB_IDS = [str(job["id"]) for job in jobs]
    return {"ok": True, "message": message, "jobs": jobs}


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_ROUTES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/static/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


def load_static_asset(route: str) -> tuple[bytes, str] | None:
    asset = STATIC_ROUTES.get(route)
    if asset is None:
        return None
    filename, content_type = asset
    asset_path = STATIC_DIR / filename
    if not asset_path.is_file():
        return None
    return asset_path.read_bytes(), content_type


def static_response(handler: BaseHTTPRequestHandler, body: bytes, content_type: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(body)


class WebAgentHandler(BaseHTTPRequestHandler):
    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if os.getenv("AGENT_WEB_LOG_REQUESTS", "").strip().lower() in {"1", "true", "yes"}:
            super().log_request(code, size)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[WebAgent] " + fmt % args + "\n")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        static_asset = load_static_asset(path)
        if static_asset is not None:
            body, content_type = static_asset
            static_response(self, body, content_type)
            return
        if path == "/api/bootstrap":
            json_response(self, {
                "ok": True,
                "tasks": task_options(),
                "actions": [
                    {"name": k, "title": v.title, "description": v.description}
                    for k, v in ACTIONS.items()
                ],
                "jobs": [asdict(job) for job in JOBS.values()],
            })
            return
        json_response(self, {"ok": False, "error": "Not found"}, 404)

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            payload = read_json(self)
            if path == "/api/intent":
                text = str(payload.get("text") or "")
                result = classify_text(text)
                form = form_for_action(result["action"], result.get("parameters") or {})
                json_response(self, {**result, "ok": True, "form": form})
                return
            if path == "/api/form":
                action = str(payload.get("action") or "help")
                form = form_for_action(action, dict(payload.get("params") or {}))
                json_response(self, {"ok": True, "action": action, "form": form})
                return
            if path == "/api/launch":
                result = launch_action(str(payload.get("action") or ""), dict(payload.get("params") or {}))
                json_response(self, result)
                return
            if path == "/api/stop":
                json_response(self, stop_job(payload.get("job_id")))
                return
            if path == "/api/progress":
                json_response(self, progress_for_job(str(payload.get("job_id") or "")))
                return
            if path == "/api/augment-stats":
                json_response(self, augment_stats_for_job(str(payload.get("job_id") or "")))
                return
            if path == "/api/inference-stats":
                json_response(self, inference_stats_for_job(str(payload.get("job_id") or "")))
                return
            if path == "/api/convert-progress":
                json_response(self, convert_progress_for_job(str(payload.get("job_id") or "")))
                return
            if path == "/api/hunyuan3d/submit":
                json_response(self, {"ok": True, "job": submit_hunyuan3d_job(payload)})
                return
            if path == "/api/hunyuan3d/status":
                generation_id = str(payload.get("generation_id") or "")
                json_response(self, {"ok": True, "job": hunyuan3d_job_status(generation_id)})
                return
            if path == "/api/hunyuan3d/recover":
                json_response(self, {"ok": True, "job": recover_hunyuan3d_job(payload)})
                return
            if path == "/api/hunyuan3d/assets":
                json_response(self, hunyuan3d_assets())
                return
            if path == "/api/hunyuan3d/physics":
                json_response(self, create_hunyuan3d_physics_asset(payload))
                return
            json_response(self, {"ok": False, "error": "Not found"}, 404)
        except Exception as exc:
            json_response(self, {"ok": False, "error": str(exc)}, 400)


def main() -> None:
    load_local_config()
    host = os.getenv("AGENT_WEB_HOST", "127.0.0.1")
    port = int(os.getenv("AGENT_WEB_PORT", "7860"))
    server = ThreadingHTTPServer((host, port), WebAgentHandler)
    print(f"Web Agent 已启动: http://{host}:{port}")
    print("在网页中完成问询；需要运行脚本时，Web Agent 会自动打开终端。")
    server.serve_forever()


if __name__ == "__main__":
    main()
