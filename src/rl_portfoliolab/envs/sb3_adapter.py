from __future__ import annotations

from typing import Any, Optional

from rl_portfoliolab.envs.portfolio_env import PortfolioAllocationEnv


def make_gymnasium_env(*, base_env: PortfolioAllocationEnv, gym: Any) -> Any:
    """
    Create a real `gymnasium.Env` instance wrapping `PortfolioAllocationEnv`.

    SB3 validates the environment type and expects an actual Gymnasium Env subclass.
    We define the subclass dynamically using the provided `gymnasium` module.
    """
    # gymnasium spaces use numpy dtypes; import numpy only when training runtime supports it.
    import numpy as np  # type: ignore

    n = base_env.n_assets
    obs_dim = 3 * n + 1  # [returns(N), vol(N), weights(N), equity(1)]

    class _Env(gym.Env):  # type: ignore[misc]
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.base_env = base_env
            self.observation_space = gym.spaces.Box(
                low=-1e9, high=1e9, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            obs = self.base_env.reset()
            flat = self._flatten_obs(obs)
            return np.asarray(flat, dtype=np.float32), {}

        def step(self, action):
            action_list = [float(x) for x in action]
            obs, reward, done, info = self.base_env.step(action_list)
            terminated = bool(done)
            truncated = False
            flat = self._flatten_obs(obs)
            return np.asarray(flat, dtype=np.float32), float(reward), terminated, truncated, info

        def _flatten_obs(self, obs: dict) -> list[float]:
            feats = obs["features"]
            returns = feats["returns"]
            vol = feats["volatility"]
            weights = obs["portfolio"]["weights"]
            equity = obs["portfolio"]["equity"]

            def _to_float(x):
                if x is None:
                    return 0.0
                try:
                    return float(x)
                except Exception:
                    return 0.0

            out: list[float] = []
            out.extend(_to_float(x) for x in returns)
            out.extend(_to_float(x) for x in vol)
            out.extend(_to_float(x) for x in weights)
            out.append(_to_float(equity))
            return out

    return _Env()

