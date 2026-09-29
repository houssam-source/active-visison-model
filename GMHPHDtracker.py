import numpy as np

class GMPHDTracker:
    def __init__(self, dt, process_noise_std, dist_noise_std, bear_noise_std, 
                 prune_thresh=1e-4, merge_thresh=0.5, max_components=100):
        # System constants
        self.dt = dt
        self.max_components = max_components
        
        # Thresholds for management
        self.prune_thresh = prune_thresh
        self.merge_thresh = merge_thresh
        
        # Noise Matrices (Eq. 9)
        self.Qt = self._build_process_noise(process_noise_std, dt)
        self.Rt = np.diag([dist_noise_std**2, bear_noise_std**2])
        
        # State Transition Matrices (Eq. 6)
        self.Ft = np.block([[np.eye(2), dt * np.eye(2)], 
                            [np.zeros((2, 2)), np.eye(2)]])
        # Ct maps 2D ego-velocity to 4D state (affects position only)
        self.Ct = np.vstack([dt * np.eye(2), np.zeros((2, 2))]) 
        
        # The Gaussian Mixture (List of dicts)
        self.gaussians = []

    def _build_process_noise(self, sigma_v, dt):
        t4 = (dt**4) / 4
        t3 = (dt**3) / 2
        t2 = dt**2
        q = sigma_v**2
        Qt = np.array([[t4*q, 0, t3*q, 0],
                       [0, t4*q, 0, t3*q],
                       [t3*q, 0, t2*q, 0],
                       [0, t3*q, 0, t2*q]])
        return Qt

    # ---------------------------------------------------------
    # PHASE A: PREDICTION (Time Update)
    # ---------------------------------------------------------
    def predict(self, ego_velocity):
        """
        Predicts the next state for all existing Gaussians.
        ego_velocity: 2x1 array [v_x_uav, v_y_uav] (Your drone's velocity)
        """
        ego_velocity = np.asarray(ego_velocity, dtype=float).reshape(-1)
        if ego_velocity.size != 2 or not np.isfinite(ego_velocity).all():
            raise ValueError("ego_velocity must contain two finite values")
        u_t = self.Ct @ ego_velocity

        for g in self.gaussians:
            # Eq. 5: propagate each Gaussian mean and covariance forward in time.
            g['mean'] = self.Ft @ g['mean'] + u_t
            g['cov'] = self.Ft @ g['cov'] @ self.Ft.T + self.Qt

    # ---------------------------------------------------------
    # PHASE B: UPDATE (Measurement Update / EKF)
    # ---------------------------------------------------------
    def _get_jacobian(self, mean):
        """
        Calculates the Jacobian matrix Ht (Eq. 8) for a given state mean.
        """
        px, py = mean[0], mean[1]
        d_sq = px**2 + py**2
        d = np.sqrt(d_sq)
        
        if d < 1e-6:
            return np.zeros((2, 4)) # Avoid division by zero
            
        Ht = np.zeros((2, 4))
        # Jacobian for the polar observation model.
        # Row 0: derivatives for distance (d)
        # Row 1: derivatives for bearing (beta)
        jacobian_ht = np.array([[px/d, py/d, 0, 0],
                                [-py/d_sq, px/d_sq, 0, 0]])
        Ht = jacobian_ht
        
        pass
        return Ht

    def _observation_model(self, mean):
        """
        Converts Cartesian state to Polar measurement (Eq. 7).
        """
        px, py = mean[0], mean[1]
        d = np.sqrt(px**2 + py**2)
        beta = np.arctan2(py, px)
        return np.array([d, beta])

    def update(self, measurements):
        """
        Updates the Gaussian mixture using new polar measurements [d, beta].
        measurements: List of 2x1 arrays.
        """
        if len(measurements) == 0:
            return

        measurements = [np.asarray(m, dtype=float).reshape(2) for m in measurements]
        if any(not np.isfinite(z).all() or z[0] <= 0 for z in measurements):
            raise ValueError("Measurements must contain finite values and positive ranges")
        new_gaussians = []
        matched_measurements = set()

        # Track which measurement has already been explained by a Gaussian.
        # Without this check, the same observation can be re-used as a birth candidate
        # on every iteration, causing duplicate newborn components and unstable PHD growth.
        for g in self.gaussians:
            if np.linalg.norm(g['mean'][:2]) < 1e-6:
                continue
            for idx, z in enumerate(measurements):
                Ht = self._get_jacobian(g['mean'])
                z_pred = self._observation_model(g['mean'])

                # Innovation (Residual)
                y = z - z_pred
                # Normalize bearing angle to [-pi, pi]
                y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi

                # Innovation Covariance
                St = Ht @ g['cov'] @ Ht.T + self.Rt
                St = 0.5 * (St + St.T)
                sign, logdet = np.linalg.slogdet(St)
                if sign <= 0:
                    continue

                try:
                    solved_innovation = np.linalg.solve(St, y)
                    solved_cross_covariance = np.linalg.solve(St, Ht @ g['cov'])
                except np.linalg.LinAlgError:
                    continue

                mahalanobis_sq = y.T @ solved_innovation
                gaussian_likelihood = np.exp(
                    -0.5 * (mahalanobis_sq + len(z) * np.log(2 * np.pi) + logdet)
                )

                # Only count a measurement as matched when it is plausible under the current Gaussian.
                if gaussian_likelihood > 1e-8:
                    matched_measurements.add(idx)

                # Standard EKF update
                Kt = solved_cross_covariance.T
                new_mean = g['mean'] + Kt @ y
                identity_minus_kh = np.eye(4) - Kt @ Ht
                new_cov = (
                    identity_minus_kh @ g['cov'] @ identity_minus_kh.T
                    + Kt @ self.Rt @ Kt.T
                )
                new_cov = 0.5 * (new_cov + new_cov.T)

                new_gaussians.append({
                    'mean': new_mean,
                    'cov': new_cov,
                    'weight': g['weight'] * gaussian_likelihood,
                })

        # 2. Birth new Gaussians for unassociated measurements.
        for idx, z in enumerate(measurements):
            if idx in matched_measurements:
                continue

            d, beta = z
            px = d * np.cos(beta)
            py = d * np.sin(beta)
            new_gaussians.append({
                'mean': np.array([px, py, 0.0, 0.0]),
                'cov': np.diag([1.0, 1.0, 10.0, 10.0]),
                'weight': 1.0,
            })

        self.gaussians = new_gaussians
        self._prune_and_merge()

    # ---------------------------------------------------------
    # PHASE C: MANAGEMENT (Pruning & Merging)
    # ---------------------------------------------------------
    def _prune_and_merge(self):
        """
        Removes weak Gaussians and merges close ones to prevent exponential growth.
        """
        # 1. Pruning
        self.gaussians = [g for g in self.gaussians if g['weight'] > self.prune_thresh]

        # 2. Merging (Simplified)
        # Merge Gaussians that are spatially close so the mixture does not explode
        # with many nearly identical components tracking the same target.
        merged = []
        used = set()

        for i, g in enumerate(self.gaussians):
            if i in used:
                continue
            merged_g = g.copy()

            for j in range(i + 1, len(self.gaussians)):
                if j in used:
                    continue
                h = self.gaussians[j]
                dist = np.linalg.norm(merged_g['mean'][:2] - h['mean'][:2])

                if dist < self.merge_thresh:
                    weight_g = merged_g['weight']
                    weight_h = h['weight']
                    total_weight = weight_g + weight_h
                    mean_g = merged_g['mean']
                    mean_h = h['mean']
                    mean = (weight_g * mean_g + weight_h * mean_h) / total_weight
                    delta_g = mean_g - mean
                    delta_h = mean_h - mean
                    merged_g['cov'] = (
                        weight_g * (merged_g['cov'] + np.outer(delta_g, delta_g))
                        + weight_h * (h['cov'] + np.outer(delta_h, delta_h))
                    ) / total_weight
                    merged_g['mean'] = mean
                    merged_g['weight'] = total_weight
                    used.add(j)

            merged.append(merged_g)

        self.gaussians = merged

        # 3. Cap maximum components
        if len(self.gaussians) > self.max_components:
            # Sort by weight and keep top N
            self.gaussians.sort(key=lambda g: g['weight'], reverse=True)
            self.gaussians = self.gaussians[:self.max_components]