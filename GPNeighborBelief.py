import numpy as np

class GPNeighborBelief:
    def __init__(self, l_spatial, l_temporal, sigma_f, noise_std):
        self.X_hist = np.empty((0, 3))  # [x, y, t]
        self.Z_hist = np.empty((0, 1))  # [0 or 1]
        self.l1, self.l2 = l_spatial
        self.l3 = l_temporal
        self.sigma_f = sigma_f
        self.noise_std = noise_std
        self.prune_thresh = 1.993 * l_temporal

    def _matern32_kernel(self, X1, X2):
        """
        Computes the Matérn 3/2 kernel matrix between X1 (Nx3) and X2 (Mx3).
        Output shape: (N, M)
        """
        if X1.size == 0 or X2.size == 0:
            return np.empty((len(X1), len(X2)))

        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        diff = X1[:, None, :] - X2[None, :, :]
        scaled = diff / np.array([self.l1, self.l2, self.l3], dtype=float)
        r2 = np.sum(scaled**2, axis=2)
        r = np.sqrt(r2)

        # Matérn 3/2 kernel: k(r) = sigma_f^2 * (1 + sqrt(3) r) * exp(-sqrt(3) r)
        sqrt3 = np.sqrt(3.0)
        K = self.sigma_f**2 * (1.0 + sqrt3 * r) * np.exp(-sqrt3 * r)
        return K

    def prune_old_data(self, current_time):
        """Remove historical points older than prune_thresh."""
        # TODO: Filter self.X_hist and self.Z_hist
        if self.X_hist.size > 0:
            time_diffs = current_time - self.X_hist[:, 2]
            mask = time_diffs <= self.prune_thresh
            self.X_hist = self.X_hist[mask]
            self.Z_hist = self.Z_hist[mask]
        pass

    def update(self, x, y, t, is_valid):
        """Append new measurement to history."""
        # TODO: Stack new point, call prune_old_data
        new_point = np.array([[x, y, t]], dtype=float)
        new_label = np.array([[1.0 if is_valid else 0.0]], dtype=float)
        self.X_hist = np.vstack([self.X_hist, new_point])
        self.Z_hist = np.vstack([self.Z_hist, new_label])
        return self.prune_old_data(t)

    def predict(self, x_q, y_q, t_q):
        # 1. Empty history guard (returns prior)
        if self.X_hist.shape[0] == 0:
            return 0.0, self.sigma_f

        X_new = np.array([[x_q, y_q, t_q]])
        K = self._matern32_kernel(self.X_hist, self.X_hist)
        K_s = self._matern32_kernel(self.X_hist, X_new)   # (N, 1)
        K_ss = self._matern32_kernel(X_new, X_new)        # (1, 1)

        # Add measurement noise to training covariance
        K_noisy = K + (self.noise_std**2) * np.eye(len(self.X_hist))

        # 2. Use SOLVE instead of INV for numerical stability
        try:
            v = np.linalg.solve(K_noisy, K_s)           # v = K^-1 @ K_s
            alpha = np.linalg.solve(K_noisy, self.Z_hist) # alpha = K^-1 @ Z
        except np.linalg.LinAlgError:
            # Fallback for ill-conditioned matrices
            reg = K_noisy + 1e-6 * np.eye(len(K_noisy))
            v = np.linalg.solve(reg, K_s)
            alpha = np.linalg.solve(reg, self.Z_hist)

        # 3. Posterior mean & variance
        mu = (K_s.T @ alpha).flatten()[0]
        var = K_ss.flatten()[0] - (K_s.T @ v).flatten()[0]

        # 4. Clamp negative variance from floating-point errors
        return mu, np.sqrt(max(var, 0.0))
class UncertaintyQuantifier:
    def __init__(self, x_range=(-10, 10), y_range=(-10, 10), resolution=0.5, max_fov_range=8.0):
        # 1. Create coordinate grids
        self.x = np.arange(x_range[0], x_range[1], resolution)
        self.y = np.arange(y_range[0], y_range[1], resolution)
        self.X_grid, self.Y_grid = np.meshgrid(self.x, self.y)
        self.grid_shape = self.X_grid.shape
        self.max_fov_range = max_fov_range

    def _generate_fov_mask(self, uav_heading):
        """
        Vectorized binary mask for 90° FOV (±45° from heading).
        Assumes UAV is at relative origin (0,0).
        """
        # Relative angles to all grid cells
        angles = np.arctan2(self.Y_grid, self.X_grid)
        rel_angles = angles - uav_heading

        # Normalize to [-pi, pi]
        rel_angles = (rel_angles + np.pi) % (2 * np.pi) - np.pi

        # Inside cone? (90° total = ±45° = ±π/4)
        in_cone = np.abs(rel_angles) <= (np.pi / 4)

        # Within sensor range?
        in_range = np.hypot(self.X_grid, self.Y_grid) <= self.max_fov_range

        return in_cone & in_range

    def compute_UNC_matrix(self, gps_list, uav_heading, t_now):
        """
        Algorithm 1: Computes global UNC matrix U(x,y).
        """
        # 1. Initialize U to maximum uncertainty (1.0)
        U = np.ones(self.grid_shape)

        # Flatten grid for efficient GP querying
        x_flat = self.X_grid.ravel()
        y_flat = self.Y_grid.ravel()

        for gp in gps_list:
            # Prune stale data
            gp.prune_old_data(t_now)

            # 2. Query GP uncertainty (sigma) over entire grid
            # [predict returns (mu, sigma), we only need sigma]
            sigma_flat = np.array([gp.predict(x, y, t_now)[1] for x, y in zip(x_flat, y_flat)])
            sigma_grid = sigma_flat.reshape(self.grid_shape)

            # 3. Generate current FOV mask
            mask = self._generate_fov_mask(uav_heading)

            # 4. Apply FOV Mask: U = min(U, σ) where mask==1, else keep U
            U = np.where(mask, np.minimum(U, sigma_grid), U)

        return U
