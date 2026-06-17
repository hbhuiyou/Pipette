from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Sequence


def format_command(argv: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(x) for x in argv])
    return " ".join(shlex.quote(str(x)) for x in argv)


def launch_new_terminal(
    argv: Sequence[str],
    cwd: Path,
    title: str,
    pid_file: Path | None = None,
    keep_open_default: bool = False,
    env_unset: Sequence[str] | None = None,
    log_file: Path | None = None,
) -> subprocess.Popen:
    cwd = Path(cwd).resolve()
    command_text = format_command([str(x) for x in argv])
    pid_file = Path(pid_file).resolve() if pid_file is not None else None
    log_file = Path(log_file).resolve() if log_file is not None else None
    env_unset = tuple(env_unset or ())
    keep_open_env = os.getenv("AGENT_KEEP_TERMINAL_OPEN", "").strip().lower()
    keep_open = bool(keep_open_default)
    if keep_open_env:
        keep_open = keep_open_env in {"1", "true", "yes"}

    if os.name == "nt":
        ps_command = f"$Host.UI.RawUI.WindowTitle = {ps_quote(title)}; Set-Location -LiteralPath {ps_quote(str(cwd))}; "
        if pid_file is not None:
            ps_command += f"$PID | Set-Content -LiteralPath {ps_quote(str(pid_file))}; "
        for name in env_unset:
            ps_command += f"Remove-Item Env:{name} -ErrorAction SilentlyContinue; "
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            ps_command += "$env:PYTHONUNBUFFERED='1'; "
            ps_command += f"& {command_text} 2>&1 | Tee-Object -FilePath {ps_quote(str(log_file))} -Append"
        else:
            ps_command += f"& {command_text}"
        if keep_open:
            ps_command += "; Write-Host ''; Write-Host '任务进程已结束。可以查看上方输出，确认无误后关闭窗口。'"
        powershell_args = [
            "powershell.exe",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps_command,
        ]
        if keep_open:
            powershell_args.insert(1, "-NoExit")
        return subprocess.Popen(
            powershell_args,
            cwd=str(cwd),
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )

    terminal_candidates = [
        ["gnome-terminal", "--"],
        ["konsole", "-e"],
        ["xterm", "-e"],
    ]
    shell_prefix = ""
    if pid_file is not None:
        shell_prefix = f"echo $$ > {shlex.quote(str(pid_file))}; "
    if env_unset:
        shell_prefix += "unset " + " ".join(shlex.quote(name) for name in env_unset) + "; "
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        shell_prefix += f"set -o pipefail; export PYTHONUNBUFFERED=1; : > {shlex.quote(str(log_file))}; "
        command_body = f"{command_text} 2>&1 | tee -a {shlex.quote(str(log_file))}"
    else:
        command_body = command_text
    if keep_open:
        shell_command = f"{shell_prefix}cd {shlex.quote(str(cwd))} && {command_body}"
    else:
        if log_file is not None:
            shell_command = f"{shell_prefix}cd {shlex.quote(str(cwd))} && {command_body}"
        else:
            shell_command = f"{shell_prefix}cd {shlex.quote(str(cwd))} && exec {command_body}"
    if keep_open:
        shell_command += "; echo; read -p 'Press Enter to close...'"
    for prefix in terminal_candidates:
        try:
            return subprocess.Popen(prefix + ["bash", "-lc", shell_command], cwd=str(cwd))
        except FileNotFoundError:
            continue

    env = os.environ.copy()
    for name in env_unset:
        env.pop(name, None)
    return subprocess.Popen([str(x) for x in argv], cwd=str(cwd), env=env)


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
