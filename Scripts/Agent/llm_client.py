from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Intent:
    action: str = "help"
    confidence: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    source: str = "fallback"


class OpenAICompatibleClient:
    """Small OpenAI-compatible chat client using only the Python standard library."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else (
            os.getenv("DASHSCOPE_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
        )
        self.base_url = (
            base_url
            if base_url is not None
            else os.getenv("DASHSCOPE_BASE_URL", "")
            or os.getenv("OPENAI_BASE_URL", "")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.model = (
            model
            if model is not None
            else os.getenv("DASHSCOPE_MODEL", "")
            or os.getenv("OPENAI_MODEL", "")
            or "deepseekV4"
        )
        self.timeout = float(timeout)
        self.use_response_format = os.getenv("AGENT_USE_JSON_RESPONSE_FORMAT", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }

    @property
    def available(self) -> bool:
        return bool(self.api_key.strip())

    def classify_intent(
        self,
        user_text: str,
        actions: list[dict[str, str]],
        tasks: list[dict[str, str]],
    ) -> Intent:
        if not self.available:
            return Intent(reply="未配置 OPENAI_API_KEY，已使用本地规则理解你的意图。")

        system_prompt = (
            "你是 IsaacLab 脚本调度智能体。你只做意图识别，不编造脚本。"
            "请根据用户中文自然语言，在给定 actions 中选择最匹配的一项，"
            "并抽取可用参数。必须只输出 JSON。"
        )
        user_prompt = {
            "user_text": user_text,
            "actions": actions,
            "tasks": tasks,
            "output_schema": {
                "action": "one of actions.name",
                "confidence": "0.0 to 1.0",
                "parameters": {
                    "task_id": "optional task id",
                    "policy_type": "optional: auto, act, smolvla, pi0, vla",
                    "dataset_file": "optional",
                    "usd_path": "optional USD scene path for register_task or build_environment",
                    "asset_dir": "optional asset directory for build_environment",
                    "lab_description": "optional lab scene description for analyze_assets",
                    "description": "optional task description for register_task",
                    "language_instruction": "optional task language instruction for register_task",
                    "num_demos": "optional integer",
                    "episodes": "optional integer",
                },
                "reply": "brief Chinese explanation",
            },
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
        }
        if self.use_response_format:
            body["response_format"] = {"type": "json_object"}

        try:
            payload = json.dumps(body).encode("utf-8")
            request = urllib.request.Request(
                url=f"{self.base_url}/chat/completions",
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            parsed = _parse_json_object(content)
            return Intent(
                action=str(parsed.get("action") or "help"),
                confidence=float(parsed.get("confidence") or 0.0),
                parameters=dict(parsed.get("parameters") or {}),
                reply=str(parsed.get("reply") or ""),
                source="api",
            )
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            return Intent(reply=f"API 请求失败，已改用本地规则。HTTP {exc.code}: {details[:180]}")
        except Exception as exc:
            return Intent(reply=f"API 调用不可用，已改用本地规则。原因: {exc}")

    def generate_task_registration_fields(self, task_description: str) -> dict[str, str]:
        if not self.available:
            return {
                "task_id": "",
                "description": task_description,
                "language_instruction": "",
                "source": "fallback",
                "warning": "未配置 API Key，无法调用大模型自动生成字段。",
            }

        system_prompt = (
            "你是 IsaacLab 任务注册助手。用户会给出一个自然语言任务描述。"
            "请生成 task_id、description、language_instruction 三个字段。"
            "task_id 必须是英文小写 snake_case，只能包含 a-z、0-9、下划线，必须以字母开头。"
            "description 用英文一句话描述采集任务，推荐格式：Keyboard teleoperation data collection for xxx task."
            "language_instruction 用英文小写 snake_case 描述机器人要做什么，例如 open_the_box。"
            "不要生成 dataset_file，不要生成代码，必须只输出 JSON。"
        )
        user_prompt = {
            "task_description": task_description,
            "output_schema": {
                "task_id": "lower_snake_case English id",
                "description": "English human-readable task description",
                "language_instruction": "lower_snake_case English task instruction",
            },
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
        }
        if self.use_response_format:
            body["response_format"] = {"type": "json_object"}

        try:
            payload = json.dumps(body).encode("utf-8")
            request = urllib.request.Request(
                url=f"{self.base_url}/chat/completions",
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            parsed = _parse_json_object(content)
            return {
                "task_id": str(parsed.get("task_id") or ""),
                "description": str(parsed.get("description") or task_description),
                "language_instruction": str(parsed.get("language_instruction") or ""),
                "source": "api",
                "warning": "",
            }
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            return {
                "task_id": "",
                "description": task_description,
                "language_instruction": "",
                "source": "fallback",
                "warning": f"API 请求失败，无法自动生成字段。HTTP {exc.code}: {details[:180]}",
            }
        except Exception as exc:
            return {
                "task_id": "",
                "description": task_description,
                "language_instruction": "",
                "source": "fallback",
                "warning": f"API 调用不可用，无法自动生成字段。原因: {exc}",
            }

    def analyze_lab_assets(self, lab_description: str, owned_assets: list[str]) -> dict[str, Any]:
        """Analyze which assets are needed for a lab scene and match them to owned USD files."""
        if not self.available:
            return {
                "source": "fallback",
                "summary": "未配置 API Key，已使用本地关键词规则生成资产清单。",
                "needed_assets": [],
                "owned_assets": [],
                "missing_assets": [],
                "suggestions": ["配置 DASHSCOPE_API_KEY 后可以让大模型给出更准确的资产拆解。"],
            }

        system_prompt = (
            "你是 Isaac Sim 生物实验室场景资产规划助手。"
            "用户会给出要搭建的实验室或任务场景描述，以及当前 asset 文件夹中已有的 USD 资产文件名。"
            "请先分析这个场景通常需要哪些资产，然后把它们和已有文件进行匹配。"
            "匹配时允许同义词和大小写差异，例如 biosafety cabinet 与 safety cabinet 可以视为相关。"
            "只能输出 JSON，不要输出 Markdown。"
        )
        user_prompt = {
            "lab_description": lab_description,
            "owned_usd_assets": owned_assets,
            "output_schema": {
                "summary": "one short Chinese summary",
                "needed_assets": [
                    {
                        "name": "Chinese or English asset name",
                        "purpose": "why this asset is needed",
                        "matched_files": ["owned USD file names that can satisfy this need"],
                    }
                ],
                "owned_assets": [
                    {
                        "file": "owned USD file name",
                        "use": "how it can be used in this scene",
                    }
                ],
                "missing_assets": [
                    {
                        "name": "missing asset name",
                        "reason": "why it is missing or why existing assets are insufficient",
                    }
                ],
                "suggestions": ["short Chinese next step suggestions"],
            },
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
        }
        if self.use_response_format:
            body["response_format"] = {"type": "json_object"}

        try:
            payload = json.dumps(body).encode("utf-8")
            request = urllib.request.Request(
                url=f"{self.base_url}/chat/completions",
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            parsed = _parse_json_object(content)
            return {
                "source": "api",
                "summary": str(parsed.get("summary") or ""),
                "needed_assets": list(parsed.get("needed_assets") or []),
                "owned_assets": list(parsed.get("owned_assets") or []),
                "missing_assets": list(parsed.get("missing_assets") or []),
                "suggestions": list(parsed.get("suggestions") or []),
            }
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            return {
                "source": "fallback",
                "summary": f"API 请求失败，已改用本地规则。HTTP {exc.code}: {details[:180]}",
                "needed_assets": [],
                "owned_assets": [],
                "missing_assets": [],
                "suggestions": [],
            }
        except Exception as exc:
            return {
                "source": "fallback",
                "summary": f"API 调用不可用，已改用本地规则。原因: {exc}",
                "needed_assets": [],
                "owned_assets": [],
                "missing_assets": [],
                "suggestions": [],
            }


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def fallback_intent(user_text: str) -> Intent:
    text = user_text.strip().lower()
    if not text:
        return Intent(action="help", confidence=0.0)

    stop_keywords = (
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
    )
    if text in stop_keywords or any(keyword in text for keyword in stop_keywords):
        return Intent(action="stop_active_job", confidence=1.0)

    exit_keywords = (
        "exit",
        "quit",
        "q",
        "退出",
        "结束",
        "关闭",
        "退出脚本",
        "关闭脚本",
        "结束脚本",
    )
    if text in exit_keywords or any(keyword in text for keyword in exit_keywords if len(keyword) > 1):
        return Intent(action="exit", confidence=1.0)

    keyword_map: list[tuple[str, tuple[str, ...]]] = [
        ("analyze_assets", ("资产分析", "分析资产", "资产清单", "已有资产", "缺失资产", "缺少哪些资产", "生物实验室", "实验室资产", "lab assets")),
        ("collect_data", ("采集", "收集", "遥操作", "teleop", "collect", "record")),
        ("augment_data", ("增强", "增广", "augmentation", "augment", "光照", "扰动")),
        ("replay_data", ("回放", "重放", "replay", "播放轨迹")),
        ("inspect_dataset", ("查看", "检查", "统计", "inspect", "数据集信息", "结构")),
        ("convert_lerobot", ("lerobot", "转换", "convert")),
        ("train_model", ("训练", "train", "finetune", "fine-tune", "微调", "训练模型")),
        ("delete_demo", ("删除", "delete", "移除", "清理")),
        ("register_task", ("注册任务", "添加任务", "新增任务", "新建任务", "登记任务", "task_registry", "register task")),
        ("build_environment", ("搭建环境", "搭建场景", "编辑环境", "编辑场景", "打开环境", "打开场景", "修改环境", "修改场景", "build environment", "edit usd")),
        (
            "run_inference",
            (
                "推理",
                "评估",
                "运行模型",
                "启动模型",
                "模型服务",
                "加载模型",
                "服务",
                "服务器",
                "server",
                "policy-path",
                "inference",
                "evaluate",
                "eval",
            ),
        ),
    ]
    for action, keywords in keyword_map:
        if any(keyword in text for keyword in keywords):
            return Intent(action=action, confidence=0.55)

    return Intent(action="help", confidence=0.2)
