#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


IMAGE = "gitlab-registry.cern.ch/atlas/statanalysis:0-4"
CONTAINER_ENGINE = "podman-hpc"

PROJECT_DIR = Path.cwd().resolve()
CONTAINER_PROJECT_DIR = "/workdir"


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def to_container_path(path: str | Path) -> str:
    """
    Convert a project-local path to the corresponding path inside the container.

    Example:
      configs/myfit.config -> /workdir/configs/myfit.config
    """
    path = Path(path)

    if path.is_absolute():
        path = path.resolve()
        try:
            rel = path.relative_to(PROJECT_DIR)
        except ValueError as exc:
            raise ValueError(
                f"Path {path} is outside PROJECT_DIR={PROJECT_DIR}. "
                "Move it under the project directory or add another mount."
            ) from exc
    else:
        rel = path

    return str(Path(CONTAINER_PROJECT_DIR) / rel)


def container_cmd(shell_command: str) -> list[str]:
    """
    Build a podman-hpc command for Perlmutter.

    Important:
      - no -it from Python
      - --group-add keep-groups is needed for project directory permissions
      - PROJECT_DIR is mounted as /workdir
    """
    return [
        CONTAINER_ENGINE,
        "run",
        "--rm",
        "--group-add",
        "keep-groups",
        "-v",
        f"{PROJECT_DIR}:{CONTAINER_PROJECT_DIR}",
        "-w",
        CONTAINER_PROJECT_DIR,
        IMAGE,
        "bash",
        "-lc",
        shell_command,
    ]


def run(cmd: list[str], label: str, log_dir: Path | None = None) -> None:
    print(f"\n=== {label} ===")
    print(quote_cmd(cmd))

    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "_")
        (log_dir / f"{safe_label}.stdout.log").write_text(result.stdout)
        (log_dir / f"{safe_label}.stderr.log").write_text(result.stderr)

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")

    if result.stderr:
        print(result.stderr)


def check_setup() -> None:
    shell_command = (
        "echo 'Container:' && hostname && "
        "echo 'Working directory:' && pwd && "
        "echo 'User/groups:' && id && "
        "echo 'trex-fitter path:' && which trex-fitter && "
        "echo 'Files in /workdir:' && ls -la /workdir"
    )

    run(container_cmd(shell_command), label="check setup")


def run_trex_action(
    action: str,
    config: str | Path,
    log_dir: str | Path = "trex_logs",
) -> None:
    config_in_container = to_container_path(config)

    shell_command = (
        f"trex-fitter {shlex.quote(action)} {shlex.quote(config_in_container)}"
    )

    run(
        container_cmd(shell_command),
        label=f"trex-fitter {action}",
        log_dir=Path(log_dir),
    )


def run_trex_sequence(
    actions: list[str],
    config: str | Path,
    log_dir: str | Path = "trex_logs",
) -> None:
    for action in actions:
        run_trex_action(action, config, log_dir=log_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preinstalled TRExFitter from StatAnalysis image using podman-hpc."
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="configs/myfit.config",
        help="TRExFitter config file relative to the project directory.",
    )

    parser.add_argument(
        "--actions",
        nargs="+",
        default=["h", "w", "f"],
        help="TRExFitter actions to run, e.g. --actions h w f s l d",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check container setup and exit.",
    )

    parser.add_argument(
        "--log-dir",
        default="trex_logs",
        help="Directory where stdout/stderr logs are saved.",
    )

    args = parser.parse_args()

    if args.check:
        check_setup()
        return

    run_trex_sequence(args.actions, args.config, log_dir=args.log_dir)


if __name__ == "__main__":
    main()