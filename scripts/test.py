#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from agent import AnalysisEnv
from agent.Actions import Action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a baseline-first rollout through AnalysisEnv."
    )
    parser.add_argument(
        "--config",
        default=PROJECT_DIR / "configs" / "hyy.config",
        type=Path,
        help="Starting TRExFitter config.",
    )
    parser.add_argument(
        "--iterations",
        default=3,
        type=int,
        help="Number of sampled actions to apply after the baseline run.",
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Episode directory name under runs/. Defaults to a random id.",
    )
    parser.add_argument(
        "--workdir",
        default=PROJECT_DIR / "runs",
        type=Path,
        help="Directory where rollout artifacts are written.",
    )
    parser.add_argument(
        "--target-significance",
        default=3.0,
        type=float,
        help="Reward target used by AnalysisEnv.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use scripts/mock_trex.py instead of the real scripts/trex.py runner.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="Optional random seed for action sampling.",
    )
    return parser.parse_args()


def run_baseline(env: AnalysisEnv) -> tuple[float, dict]:
    assert env._episode_dir is not None

    baseline_config = env._episode_dir / "step_000.config"
    baseline_config.write_text("\n".join(env._lines) + "\n", encoding="utf-8")
    (env._episode_dir / "step_000.action.json").write_text(
        json.dumps(
            {
                "step": 0,
                "config": str(baseline_config),
                "action": None,
                "edit": None,
                "edit_error": None,
                "label": "baseline",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fit_result = env._run_trex(baseline_config)
    env._fit_result = fit_result
    baseline_significance = env._current_significance()
    env._best_significance = baseline_significance
    env._best_config_text = "\n".join(env._lines) + "\n"
    (env._episode_dir / "best.config").write_text(env._best_config_text, encoding="utf-8")

    return baseline_significance, env._get_info()


def format_status(info: dict) -> str:
    if "edit_error" in info:
        return f"not valid: {info['edit_error']}"
    if not info["last_success"]:
        return f"runner failed: returncode={info['last_returncode']}"
    return "valid"


def main() -> None:
    args = parse_args()
    episode_id = args.episode_id or f"sampled_actions_{uuid.uuid4().hex[:8]}"
    runner = PROJECT_DIR / "scripts" / ("mock_trex.py" if args.test else "trex.py")

    env = AnalysisEnv(
        config_path=args.config,
        trex_runner=runner,
        workdir=args.workdir,
        max_steps=args.iterations,
        target_significance=args.target_significance,
    )
    env.reset(seed=args.seed, options={"episode_id": episode_id})

    print(f"episode: {episode_id}")
    print(f"runner: {runner.relative_to(PROJECT_DIR)}")

    baseline_significance, baseline_info = run_baseline(env)
    print(
        "step=000 status=baseline "
        f"significance={baseline_significance:.6f} "
        f"reward={baseline_significance - args.target_significance:.6f}"
    )

    last_info = baseline_info
    for _ in range(args.iterations):
        action = Action.from_dict(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)
        last_info = info

        print(
            f"step={int(obs['step'][0]):03d} status={format_status(info)} "
            f"action={action} "
            f"significance={float(obs['significance'][0]):.6f} "
            f"reward={float(reward):.6f} "
            f"best={float(info['best_significance']):.6f}"
        )

        if terminated or truncated:
            break

    print(f"saved configs in: {last_info['episode_dir']}")


if __name__ == "__main__":
    main()
