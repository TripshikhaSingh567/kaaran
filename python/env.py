"""
Gymnasium Environment wrapper for the KAARAN Heated Tank Simulator.
Carries latent confounder U and disturbances in `info` for evaluation/scoring only.
"""

from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from python.sim import HeatedTankSimulator


class HeatedTankEnv(gym.Env):
    """
    Gymnasium environment for the continuous heated tank.
    Observation space: Box(low=[0, 0], high=[120, 100], dtype=float32) -> (T, L)
    Action space: Box(low=[0, 0], high=[100, 1], dtype=float32) -> (v, h)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        simulator: Optional[HeatedTankSimulator] = None,
        max_episode_steps: int = 120,
    ):
        super().__init__()
        self.sim = simulator if simulator is not None else HeatedTankSimulator()
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.state = (45.0, 55.0)

        # Observation: [Temperature, Level]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([120.0, 100.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: [Valve (0-100), Heater (0-1)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([100.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0

        # Start states: T0 ~ U(30, 60), L0 ~ U(30, 80)
        if options and "initial_state" in options:
            self.state = (float(options["initial_state"][0]), float(options["initial_state"][1]))
        else:
            T0 = float(self.np_random.uniform(30.0, 60.0))
            L0 = float(self.np_random.uniform(30.0, 80.0))
            self.state = (T0, L0)

        obs = np.array(self.state, dtype=np.float32)
        info = {
            "initial_state": self.state,
            "step": self.current_step,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
        U: Optional[float] = None,
        eps_T: Optional[float] = None,
        eps_L: Optional[float] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        v = float(action[0])
        h = float(action[1])

        # Generate latents if not explicitly passed
        latent_U = float(self.np_random.normal(0.0, 1.0)) if U is None else float(U)
        noise_T = float(self.np_random.normal(0.0, 1.0)) if eps_T is None else float(eps_T)
        noise_L = float(self.np_random.normal(0.0, 1.0)) if eps_L is None else float(eps_L)

        # Step simulator
        next_state, components = self.sim.step(
            s=self.state,
            a=(v, h),
            U=latent_U,
            eps_T=noise_T,
            eps_L=noise_L,
        )
        self.state = next_state

        reward = -abs(self.state[0] - 55.0) - 0.5 * abs(self.state[1] - 60.0)
        terminated = False
        truncated = self.current_step >= self.max_episode_steps

        # Info carries latents for evaluation/scoring only (never shown to model)
        info = {
            "U": latent_U,
            "eps_T": noise_T,
            "eps_L": noise_L,
            "components": components,
            "step": self.current_step,
        }

        obs = np.array(self.state, dtype=np.float32)
        return obs, reward, terminated, truncated, info
