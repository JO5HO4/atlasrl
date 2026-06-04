#!/usr/bin/env python3

from __future__ import annotations

import sys
import uuid
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from agent import AnalysisEnv
from agent.Actions import Action


def main() -> None:
    episode_id = f"random_rollout_{uuid.uuid4().hex[:8]}"
    env = AnalysisEnv(
        config_path=PROJECT_DIR / "configs" / "hyy.config",
        trex_runner=PROJECT_DIR / "scripts" / "mock_trex.py",
        workdir=PROJECT_DIR / "runs",
        max_steps=10,
    )
    env.reset(options={"episode_id": episode_id})

    print(f"episode: {episode_id}")

    for _ in range(10):
        action = Action.from_dict(env.action_space.sample())
        obs, reward, terminated, truncated, info = env.step(action)

        status = "valid"
        if "edit_error" in info:
            status = f"not valid: {info['edit_error']}"
        elif not info["last_success"]:
            status = "runner failed"

        print(
            f"step={int(obs['step'][0])} status={status} action={action} "
            f"significance={float(obs['significance'][0]):.6f} reward={float(reward):.6f}"
        )

        if terminated or truncated:
            break

    print(f"saved configs in: {info['episode_dir']}")


if __name__ == "__main__":
    main()
