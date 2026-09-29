import numpy as np

try:
    import gymnasium as gym  # type: ignore[import-not-found]
    from gymnasium import spaces  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency for RL usage
    gym = None
    spaces = None

from swarmmanager import SwarmActiveVisionEnv


if gym is not None:
    class ActiveVisionRLWrapper(gym.Env):
        metadata = {"render.modes": []}

        def __init__(self, n_drones=3, control_hz=10):
            super().__init__()
            self.n_drones = int(n_drones)
            self.control_hz = float(control_hz)
            self.env = SwarmActiveVisionEnv(self.n_drones, self.control_hz)

            grid_shape = self.env.pipelines[0]["unc"].X_grid.shape
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.n_drones, grid_shape[0], grid_shape[1], 2),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.5,
                high=1.5,
                shape=(self.n_drones, 2),
                dtype=np.float32,
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.env.step_counter = 0
            return self._get_obs(), {}

        def step(self, actions):
            actions = np.asarray(actions, dtype=np.float32)
            if actions.shape != self.action_space.shape:
                actions = np.clip(actions.reshape(self.action_space.shape), -1.5, 1.5)

            _, unc_maps = self.env.step()
            if not unc_maps:
                unc_maps = [
                    np.zeros_like(self.env.pipelines[i]["unc"].X_grid, dtype=np.float32)
                    for i in range(len(self.env.pipelines))
                ]

            info_gain = 0.0
            for i in range(len(unc_maps)):
                info_gain += float(np.sum(unc_maps[i] < 0.5))

            control_cost = float(np.sum(np.abs(actions)))
            total_reward = 0.1 * info_gain - 0.01 * control_cost

            return self._get_obs(), total_reward, False, False, {"info_gain": info_gain}

        def _get_obs(self):
            obs = []
            t_now = self.env.step_counter * getattr(self.env.scene, "dt", 0.0)

            for i in range(len(self.env.drones)):
                pipe = self.env.pipelines[i]
                unc = pipe["unc"]
                gp = pipe["gp"]
                U = unc.compute_UNC_matrix([gp], self.env.drones[i]["heading"], t_now)

                mean_grid = np.empty_like(U, dtype=np.float32)
                x_flat = unc.X_grid.ravel()
                y_flat = unc.Y_grid.ravel()
                for idx, (x, y) in enumerate(zip(x_flat, y_flat)):
                    mean_value, _ = gp.predict(float(x), float(y), t_now)
                    mean_grid.ravel()[idx] = mean_value

                obs.append(np.stack([U.astype(np.float32), mean_grid], axis=-1))

            return np.asarray(obs, dtype=np.float32)
else:
    class ActiveVisionRLWrapper:
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium is required for ActiveVisionRLWrapper. Install it with: pip install gymnasium")