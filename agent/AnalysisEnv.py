# analysis_env.py

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from agent.Actions import ACTION_NAMES, CUT_NAMES, PERCENT_VALUES, Action, apply_analysis_action
from agent.trex_editor import TrexConfigEditor


@dataclass
class FitResult:
    significance: float
    success: bool
    stdout: str
    stderr: str
    returncode: int


class AnalysisEnv(gym.Env):
    """
    Gymnasium environment for optimizing a TRExFitter analysis configuration.

    Action format:
        A Gymnasium Dict space with entries:

        {
            "action_type": int,
                Index into ACTION_NAMES. For the current version, this includes
                semantic actions such as "modify_category_cut".

            "region_index": int,
                Index of the TRExFitter Region/category to modify. The index is
                interpreted over the currently available analysis regions.

            "cut_type": int,
                Index into CUT_NAMES. Selects which cut or analysis quantity to
                modify, such as leading photon pT, subleading photon pT, photon
                eta acceptance, pTt, jet pT, m_jj, Δη_jj, or Δφ(γγ,jj).

            "percent_index": int,
                Index into PERCENT_VALUES. The selected cut is scaled
                multiplicatively by this percentage. For example, if the leading
                photon pT cut is 40 GeV and the selected percentage is +10,
                the new cut becomes 44 GeV.
        }

    Observation format:
        A Gymnasium Dict space with entries:

        {
            "config": str,
                Current TRExFitter config as text. The Text observation space
                uses a TRExFitter-compatible character set built from printable
                ASCII, common physics unicode symbols, whitespace, and all
                characters appearing in the initial config.

            "significance": np.ndarray, shape (1,), dtype float32,
                Current fitted significance or a default value before the first
                fit has been run.

            "step": np.ndarray, shape (1,), dtype int32,
                Current episode step.
        }
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str | Path = "configs/hyy.config",
        trex_runner: str | Path = "scripts/trex.py",
        workdir: str | Path = "runs",
        max_steps: int = 20,
        num_regions: int = 10,
        target_significance: float = 5.0,
        timeout: int = 1200,
        failure_significance: float = 0.0,
        trex_actions: list[str] | None = None,
    ):
        super().__init__()

        self.config_path = Path(config_path).resolve()
        self.trex_runner = Path(trex_runner).resolve()
        self.workdir = Path(workdir).resolve()

        self.max_steps = max_steps
        self.num_regions = num_regions
        self.target_significance = target_significance
        self.timeout = timeout
        self.failure_significance = failure_significance
        self.trex_actions = normalize_trex_actions(trex_actions or ["n", "w", "f", "s"])
        self.project_dir = self.trex_runner.parents[1]

        self.initial_config_text = self.config_path.read_text(encoding="utf-8")
        self.trex_charset = build_trex_charset(self.initial_config_text)
        detected_regions = len(TrexConfigEditor(self.initial_config_text).get_regions())
        self.num_regions = min(num_regions, detected_regions) if detected_regions else num_regions

        self.action_space = spaces.Dict(
            {
                "action_type": spaces.Discrete(len(ACTION_NAMES)),
                "region_index": spaces.Discrete(self.num_regions),
                "cut_type": spaces.Discrete(len(CUT_NAMES)),
                "percent_index": spaces.Discrete(len(PERCENT_VALUES)),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "config": spaces.Text(max_length=300_000, min_length=0, charset=self.trex_charset),
                "significance": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "step": spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32),
            }
        )

        self._lines: list[str] = []
        self._fit_result: Optional[FitResult] = None
        self._step_count = 0
        self._episode_dir: Optional[Path] = None
        self._best_significance = -np.inf
        self._best_config_text = self.initial_config_text

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        self._lines = self.initial_config_text.splitlines()
        self._fit_result = None
        self._step_count = 0
        self._best_significance = -np.inf
        self._best_config_text = self.initial_config_text

        self.workdir.mkdir(parents=True, exist_ok=True)
        episode_id = options.get("episode_id") if options else None
        if episode_id is None:
            episode_id = f"episode_{self.np_random.integers(0, 10**12)}"

        self._episode_dir = self.workdir / str(episode_id)
        if self._episode_dir.exists():
            shutil.rmtree(self._episode_dir)
        self._episode_dir.mkdir(parents=True)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action: Action | dict[str, Any]):
        self._step_count += 1

        edit_ok, edit_msg = self._apply_action(action)

        if not edit_ok:
            config_file = self._write_current_config(action, edit_error=edit_msg)
            fit_result = self._run_trex(config_file)
            self._fit_result = fit_result
            new_significance = self._current_significance()
            reward = new_significance - self.target_significance
            terminated = False
            truncated = self._step_count >= self.max_steps
            info = self._get_info()
            info["edit_error"] = edit_msg
            return self._get_obs(), reward, terminated, truncated, info

        config_file = self._write_current_config(action)
        fit_result = self._run_trex(config_file)
        self._fit_result = fit_result

        new_significance = self._current_significance()

        reward = new_significance - self.target_significance

        if new_significance > self._best_significance:
            self._best_significance = new_significance
            self._best_config_text = "\n".join(self._lines) + "\n"
            best_path = self._episode_dir / "best.config"
            best_path.write_text(self._best_config_text)

        terminated = new_significance >= self.target_significance
        truncated = self._step_count >= self.max_steps

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: Action | dict[str, Any]) -> tuple[bool, str]:
        self._last_edit_result = None
        try:
            action_obj = action if isinstance(action, Action) else Action.from_dict(action)
            normalized_action = action_obj.to_dict()
        except Exception as exc:
            return False, f"Malformed action: {exc}"

        if not self.action_space.contains(normalized_action):
            return False, f"Action outside action_space: {normalized_action}"

        try:
            editor = TrexConfigEditor("\n".join(self._lines) + "\n")
            result = apply_analysis_action(editor, action_obj)
            self._lines = editor.to_text().splitlines()
        except Exception as exc:
            return False, f"Could not apply action: {exc}"

        self._last_edit_result = result
        return True, "applied"

    def _write_current_config(
        self,
        action: Action | dict[str, Any],
        edit_error: str | None = None,
    ) -> Path:
        assert self._episode_dir is not None

        config_file = self._episode_dir / f"step_{self._step_count:03d}.config"
        config_text = "\n".join(self._lines) + "\n"
        config_file.write_text(config_text, encoding="utf-8")
        (self._episode_dir / f"step_{self._step_count:03d}.action.json").write_text(
            json.dumps(
                {
                    "step": self._step_count,
                    "config": str(config_file),
                    "action": _json_safe_action(action),
                    "edit": getattr(self, "_last_edit_result", None).__dict__
                    if getattr(self, "_last_edit_result", None)
                    else None,
                    "edit_error": edit_error,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        return config_file

    def _run_trex(self, config_file: Path) -> FitResult:
        assert self._episode_dir is not None

        cmd = [
            sys.executable,
            str(self.trex_runner),
            str(config_file),
            "--project-dir",
            str(self.project_dir),
            "--log-dir",
            str(self._episode_dir / "trex_logs"),
            "--actions",
            *self.trex_actions,
        ]

        try:
            proc = subprocess.run(
                cmd,
                cwd=self._episode_dir,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return FitResult(
                significance=self.failure_significance,
                success=False,
                stdout=exc.stdout or "",
                stderr=exc.stderr or f"Timeout after {self.timeout} seconds.",
                returncode=-999,
            )

        parsed_significance = self._parse_significance(
            stdout=proc.stdout,
            stderr=proc.stderr,
            run_dir=self._episode_dir,
        )

        success = proc.returncode == 0 and np.isfinite(parsed_significance)
        significance = (
            parsed_significance if np.isfinite(parsed_significance) else self.failure_significance
        )

        return FitResult(
            significance=significance,
            success=success,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )

    def _parse_significance(self, stdout: str, stderr: str, run_dir: Path) -> float:
        json_candidates = [
            run_dir / "results.json",
            run_dir / "fit_results.json",
        ]
        json_candidates.extend(run_dir.rglob("results.json"))
        json_candidates.extend(run_dir.rglob("fit_results.json"))

        for path in json_candidates:
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return float(data["significance"])
                except Exception:
                    pass

        text = stdout + "\n" + stderr
        patterns = [
            r"SIGNIFICANCE\s*=\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            r"Observed\s+significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            r"obs(?:erved)?\s+significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            r"Significance(?:\s+mu\s*=\s*\S+)?(?:\s*\([^)]*\))?\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
            r"Z0\s*[:=]\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))

        for log_file in run_dir.rglob("*.log"):
            try:
                log_text = log_file.read_text(errors="ignore")
            except Exception:
                continue
            for pattern in patterns:
                match = re.search(pattern, log_text, flags=re.IGNORECASE)
                if match:
                    return float(match.group(1))

        return -np.inf

    def _current_significance(self) -> float:
        if self._fit_result is None:
            return 0.0
        return float(self._fit_result.significance)

    def _get_obs(self) -> dict[str, Any]:
        return {
            "config": "\n".join(self._lines) + "\n",
            "significance": np.array([self._current_significance()], dtype=np.float32),
            "step": np.array([self._step_count], dtype=np.int32),
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self._step_count,
            "best_significance": self._best_significance,
            "episode_dir": str(self._episode_dir) if self._episode_dir else None,
            "last_returncode": self._fit_result.returncode if self._fit_result else None,
            "last_success": self._fit_result.success if self._fit_result else None,
            "last_significance": self._current_significance(),
            "last_edit": getattr(self, "_last_edit_result", None),
        }

    def render(self):
        print(f"Step: {self._step_count}")
        print(f"Current significance: {self._current_significance():.4f}")
        print(f"Best significance: {self._best_significance:.4f}")



import string

def build_trex_charset(initial_config_text: str = "") -> frozenset[str]:
    """
    Character set for TRExFitter config observations.

    Includes:
      - printable ASCII
      - newline/tab/carriage return
      - any extra characters already present in the starting config
    """

    printable_ascii = "".join(chr(i) for i in range(32, 127))
    whitespace = "\n\t\r"

    # Optional common unicode characters that may appear in comments/labels.
    # TRExFitter configs should ideally stay ASCII, but this prevents crashes
    # if comments contain arrows, unicode gamma, en-dashes, etc.
    common_unicode = "→←±−–—γΓσμνℓ√"

    charset = set(printable_ascii + whitespace + common_unicode)

    # Make sure the current config is always representable.
    charset.update(initial_config_text)

    return frozenset(charset)


def _json_safe_action(action: Action | dict[str, Any]) -> dict[str, int]:
    action_obj = action if isinstance(action, Action) else Action.from_dict(action)
    return action_obj.to_dict()


def normalize_trex_actions(actions: list[str]) -> list[str]:
    normalized = [str(action) for action in actions if str(action)]
    normalized = [action for action in normalized if action != "s"]
    normalized.append("s")
    return normalized
