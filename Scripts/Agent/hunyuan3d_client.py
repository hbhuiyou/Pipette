from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class TencentCloudAPIError(RuntimeError):
    pass


class Hunyuan3DClient:
    """Tencent Hunyuan 3D client using the TC3-HMAC-SHA256 signing protocol."""

    service = "ai3d"
    host = "ai3d.tencentcloudapi.com"
    endpoint = f"https://{host}"
    version = "2025-05-13"

    def __init__(
        self,
        secret_id: str | None = None,
        secret_key: str | None = None,
        session_token: str | None = None,
        region: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.secret_id = secret_id if secret_id is not None else os.getenv("TENCENTCLOUD_SECRET_ID", "")
        self.secret_key = secret_key if secret_key is not None else os.getenv("TENCENTCLOUD_SECRET_KEY", "")
        self.session_token = (
            session_token if session_token is not None else os.getenv("TENCENTCLOUD_SESSION_TOKEN", "")
        )
        self.region = region if region is not None else os.getenv("TENCENTCLOUD_REGION", "ap-guangzhou")
        self.timeout = float(timeout)

    @property
    def available(self) -> bool:
        return bool(self.secret_id.strip() and self.secret_key.strip())

    def submit_pro_job(
        self,
        *,
        prompt: str = "",
        image_base64: str = "",
        multi_view_images: list[dict[str, Any]] | None = None,
        model: str = "3.0",
        generate_type: str = "Normal",
        result_format: str = "USDZ",
        face_count: int | None = None,
        enable_pbr: bool = False,
    ) -> dict[str, Any]:
        inputs = [bool(prompt.strip()), bool(image_base64.strip())]
        if generate_type == "Sketch":
            valid_input = sum(inputs) >= 1
        else:
            valid_input = sum(inputs) == 1
        if not valid_input:
            raise ValueError("Prompt 和 ImageBase64 必须且只能填写一项。")
        if len(prompt) > 1024:
            raise ValueError("Prompt 不能超过 1024 个字符。")
        if model not in {"3.0", "3.1"}:
            raise ValueError("Model 只能选择 3.0 或 3.1。")
        if model == "3.1" and generate_type == "LowPoly":
            raise ValueError("混元生3D 3.1 不支持 LowPoly 模式。")

        normalized_views: list[dict[str, str]] = []
        seen_views: set[str] = set()
        allowed_views = {"left", "right", "back"}
        if model == "3.1":
            allowed_views.update({"top", "bottom", "left_front", "right_front"})
        for item in multi_view_images or []:
            view_type = str(item.get("ViewType") or "").strip().lower()
            view_base64 = str(item.get("ViewImageBase64") or "").strip()
            if view_type not in allowed_views:
                raise ValueError(f"模型 {model} 不支持多视图类型: {view_type or '空'}。")
            if view_type in seen_views:
                raise ValueError(f"多视图类型不能重复: {view_type}。")
            if not view_base64:
                raise ValueError(f"多视图 {view_type} 缺少图片内容。")
            seen_views.add(view_type)
            normalized_views.append(
                {
                    "ViewType": view_type,
                    "ViewImageBase64": view_base64,
                }
            )
        if normalized_views and not image_base64.strip():
            raise ValueError("使用多视图时必须选择一张主图。")
        encoded_images = [image_base64.strip(), *(
            item["ViewImageBase64"] for item in normalized_views
        )]
        estimated_raw_size = sum(
            max(0, len(value) * 3 // 4 - value[-2:].count("="))
            for value in encoded_images
            if value
        )
        if estimated_raw_size > 6 * 1024 * 1024:
            raise ValueError("主图和多视图图片原始内容合计不能超过 6 MB。")

        payload: dict[str, Any] = {
            "Model": model,
            "GenerateType": generate_type or "Normal",
            "ResultFormat": (result_format or "USDZ").upper(),
            "EnablePBR": bool(enable_pbr),
        }
        if prompt.strip():
            payload["Prompt"] = prompt.strip()
        else:
            payload["ImageBase64"] = image_base64.strip()
            if normalized_views:
                payload["MultiViewImages"] = normalized_views
        if face_count is not None:
            if not 3000 <= int(face_count) <= 1500000:
                raise ValueError("FaceCount 必须在 3000 到 1500000 之间。")
            payload["FaceCount"] = int(face_count)

        return self._request("SubmitHunyuanTo3DProJob", payload)

    def query_pro_job(self, job_id: str) -> dict[str, Any]:
        if not job_id.strip():
            raise ValueError("JobId 不能为空。")
        return self._request("QueryHunyuanTo3DProJob", {"JobId": job_id.strip()})

    def _request(
        self,
        action: str,
        payload: dict[str, Any],
        *,
        timestamp: int | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            raise ValueError(
                "未配置腾讯云密钥。请设置 TENCENTCLOUD_SECRET_ID 和 TENCENTCLOUD_SECRET_KEY。"
            )

        timestamp = int(time.time()) if timestamp is None else int(timestamp)
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        authorization = self._authorization(action, body, timestamp)
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json; charset=utf-8",
            "Host": self.host,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": self.version,
        }
        if self.region.strip():
            headers["X-TC-Region"] = self.region.strip()
        if self.session_token.strip():
            headers["X-TC-Token"] = self.session_token.strip()

        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise TencentCloudAPIError(f"腾讯云 API 请求失败，HTTP {exc.code}: {details[:500]}") from exc
        except urllib.error.URLError as exc:
            raise TencentCloudAPIError(f"无法连接腾讯云混元生3D API: {exc.reason}") from exc

        result = dict(data.get("Response") or {})
        error = result.get("Error")
        if error:
            code = error.get("Code") or "UnknownError"
            message = error.get("Message") or "腾讯云 API 返回错误"
            raise TencentCloudAPIError(f"{code}: {message}")
        return result

    def _authorization(self, action: str, body: bytes, timestamp: int) -> str:
        algorithm = "TC3-HMAC-SHA256"
        date = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).strftime("%Y-%m-%d")
        canonical_headers = (
            "content-type:application/json; charset=utf-8\n"
            f"host:{self.host}\n"
            f"x-tc-action:{action.lower()}\n"
        )
        signed_headers = "content-type;host;x-tc-action"
        hashed_payload = hashlib.sha256(body).hexdigest()
        canonical_request = "\n".join(
            ["POST", "/", "", canonical_headers, signed_headers, hashed_payload]
        )
        credential_scope = f"{date}/{self.service}/tc3_request"
        string_to_sign = "\n".join(
            [
                algorithm,
                str(timestamp),
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        secret_date = _hmac_sha256(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = _hmac_sha256(secret_date, self.service)
        secret_signing = _hmac_sha256(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing,
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return (
            f"{algorithm} Credential={self.secret_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )


def configured_ai3d_output_dir(project_root: Path) -> Path:
    configured = os.getenv("HUNYUAN3D_OUTPUT_DIR", "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_absolute() else (project_root / path).resolve()
    return project_root.resolve() / "Asset" / "AI3D"


def select_result_file(result_files: list[dict[str, Any]], result_format: str = "USDZ") -> dict[str, Any]:
    if not result_files:
        raise TencentCloudAPIError("任务已完成，但腾讯云未返回可下载的模型文件。")
    expected = result_format.upper()
    for item in result_files:
        if str(item.get("Type") or "").upper() == expected and item.get("Url"):
            return item
    raise TencentCloudAPIError(f"腾讯云任务已完成，但没有返回 {expected} 格式的下载地址。")


def sanitize_asset_name(asset_name: str, result_format: str = "USDZ") -> str:
    name = str(asset_name or "").strip()
    if not name:
        return ""

    extension = f".{(result_format or 'USDZ').lower()}"
    if name.lower().endswith(extension):
        name = name[: -len(extension)].strip()
    name = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name:
        raise ValueError("3D 资产名称不能只包含空格或文件名非法字符。")

    windows_reserved = {
        "CON", "PRN", "AUX", "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
    if name.upper() in windows_reserved:
        name = f"{name}_asset"
    return name[:100].rstrip(" .")


def build_output_filename(
    job_id: str,
    result_format: str = "USDZ",
    now: float | None = None,
    asset_name: str = "",
) -> str:
    extension = (result_format or "USDZ").lower()
    safe_asset_name = sanitize_asset_name(asset_name, result_format)
    if safe_asset_name:
        return f"{safe_asset_name}.{extension}"

    timestamp = dt.datetime.fromtimestamp(
        time.time() if now is None else now,
        tz=dt.timezone.utc,
    ).strftime("%Y%m%d_%H%M%S")
    safe_job_id = re.sub(r"[^A-Za-z0-9_-]+", "_", job_id).strip("_") or "job"
    return f"hunyuan3d_{timestamp}_{safe_job_id}.{extension}"


def available_asset_stem(output_dir: Path, asset_stem: str) -> str:
    if not (output_dir / asset_stem).exists():
        return asset_stem

    index = 2
    while True:
        candidate = f"{asset_stem}_{index}"
        if not (output_dir / candidate).exists():
            return candidate
        index += 1


def download_result_file(
    result_file: dict[str, Any],
    output_dir: Path,
    *,
    job_id: str,
    result_format: str = "USDZ",
    asset_name: str = "",
    timeout: float = 300.0,
) -> Path:
    url = str(result_file.get("Url") or "").strip()
    if not url:
        raise TencentCloudAPIError("模型结果缺少下载地址。")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise TencentCloudAPIError(f"不支持的模型下载地址: {url}")

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = build_output_filename(job_id, result_format, asset_name=asset_name)
    extension = Path(filename).suffix
    asset_stem = available_asset_stem(output_dir, Path(filename).stem)
    asset_dir = output_dir / asset_stem
    asset_dir.mkdir(parents=True, exist_ok=False)
    output_path = asset_dir / f"{asset_stem}{extension}"
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "IsaacSim-WebAgent/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temp_path.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
        temp_path.replace(output_path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
            asset_dir.rmdir()
        except Exception:
            pass
        raise
    return output_path


def list_generated_assets(output_dir: Path) -> list[dict[str, Any]]:
    if not output_dir.is_dir():
        return []
    assets: list[dict[str, Any]] = []
    for path in output_dir.rglob("*.usdz"):
        try:
            stat = path.stat()
        except OSError:
            continue
        usd_path = path.with_suffix(".usd")
        assets.append(
            {
                "name": path.name,
                "path": str(path),
                "folder": str(path.parent),
                "usd_path": str(usd_path) if usd_path.is_file() else "",
                "physics_ready": usd_path.is_file(),
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
            }
        )
    assets.sort(key=lambda item: float(item["modified_at"]), reverse=True)
    return assets


def _hmac_sha256(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()
