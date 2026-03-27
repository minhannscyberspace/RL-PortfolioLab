from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))


def main() -> None:
    _ensure_src_on_path()

    from rl_portfoliolab.pipeline.phase2_env import make_env_from_configs

    parser = argparse.ArgumentParser(description="Run Phase 2 environment smoke episode.")
    parser.add_argument("--config", required=True, help="Path to configs/phase2_env.yaml")
    parser.add_argument("--steps", required=False, type=int, default=25, help="Max steps to run")
    args = parser.parse_args()

    env = make_env_from_configs(phase2_yaml_path=args.config)
    obs = env.reset()

    n = len(obs["assets"])
    total_reward = 0.0
    for _ in range(args.steps):
        # Random long-only weights then normalize to gross exposure=1.
        raw = [random.random() for _ in range(n)]
        s = sum(raw)
        action = [x / s for x in raw]

        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Episode finished. total_reward={total_reward:.6f} equity={obs['portfolio']['equity']:.2f}")


if __name__ == "__main__":
    main()

