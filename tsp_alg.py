import numpy as np


class TSPHeadingPlanner:
    def __init__(self, num_candidates=9, search_range=np.pi/2, max_fov_range=8.0):
        if isinstance(num_candidates, bool) or not isinstance(num_candidates, (int, np.integer)) or num_candidates < 1:
            raise ValueError("num_candidates must be a positive integer")
        if not np.isfinite(search_range) or search_range < 0:
            raise ValueError("search_range must be finite and nonnegative")
        if not np.isfinite(max_fov_range) or max_fov_range < 0:
            raise ValueError("max_fov_range must be finite and nonnegative")

        self.num_candidates = num_candidates
        self.search_range = float(search_range)
        self.max_fov_range = float(max_fov_range)

    def _generate_candidates(self, current_heading):
        return current_heading + np.linspace(-self.search_range, self.search_range, self.num_candidates)

    def _compute_reward(self, U, x_grid, y_grid, heading):
        angles = np.arctan2(y_grid, x_grid) - heading
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        mask = (np.abs(angles) <= np.pi / 4) & (np.hypot(x_grid, y_grid) <= self.max_fov_range)
        return float(np.sum(U[mask]))

    def select_best_heading(self, U, x_grid, y_grid, current_heading):
        U = np.asarray(U, dtype=float)
        x_grid = np.asarray(x_grid, dtype=float)
        y_grid = np.asarray(y_grid, dtype=float)
        if U.ndim != 2 or x_grid.shape != U.shape or y_grid.shape != U.shape or U.size == 0:
            raise ValueError("U, x_grid, and y_grid must be nonempty 2D arrays with matching shapes")
        if not np.isfinite(U).all() or np.any(U < 0):
            raise ValueError("U must contain finite, nonnegative uncertainty values")
        if not np.isfinite(x_grid).all() or not np.isfinite(y_grid).all():
            raise ValueError("Grid coordinates must be finite")
        if not np.isfinite(current_heading):
            raise ValueError("current_heading must be finite")

        candidates = self._generate_candidates(current_heading)
        rewards = [self._compute_reward(U, x_grid, y_grid, h) for h in candidates]
        best_reward = max(rewards)
        best_indices = [
            index for index, reward in enumerate(rewards)
            if np.isclose(reward, best_reward, rtol=1e-12, atol=1e-12)
        ]
        best_idx = min(
            best_indices,
            key=lambda index: abs((candidates[index] - current_heading + np.pi) % (2 * np.pi) - np.pi),
        )
        best_heading = (candidates[best_idx] + np.pi) % (2 * np.pi) - np.pi
        return float(best_heading), best_reward