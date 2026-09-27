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
        # TODO: Construct Qt (4x4) and Rt (2x2) based on Eq. 9
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
        # TODO: Implement Eq. 9 for Qt
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
        u_t = self.Ct @ ego_velocity
        
        for g in self.gaussians:
            # TODO: Implement Eq. 5
            # g['mean'] = ...
            # g['cov'] = ...
            ego_velocity = np.asarray(ego_velocity).flatten()  # Ensure it's a column vector
            g['mean'] = self.Ft @ g['mean'] + u_t
            g['cov'] = self.Ft @ g['cov'] @ self.Ft.T + self.Qt
            pass

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
        # TODO: Fill in the 4 values for Ht based on Eq. 8
        # Row 0: derivatives for distance (d)
        # Row 1: derivatives for bearing (beta)
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
        new_gaussians = []
        
        # 1. Update existing Gaussians with each measurement
        for g in self.gaussians:
            for z in measurements:
                Ht = self._get_jacobian(g['mean'])
                z_pred = self._observation_model(g['mean'])
                
                # Innovation (Residual)
                y = z - z_pred
                # Normalize bearing angle to [-pi, pi]
                y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi 
                
                # Innovation Covariance
                St = Ht @ g['cov'] @ Ht.T + self.Rt
                
                # Kalman Gain
                Kt = g['cov'] @ Ht.T @ np.linalg.inv(St)
                
                # TODO: Calculate updated mean and cov (Standard EKF equations)
                # new_mean = ...
                # new_cov = ...
                
                # Calculate weight update (likelihood)
                # TODO: Calculate the Gaussian likelihood of the measurement
                
                new_gaussians.append({
                    'mean': new_mean,
                    'cov': new_cov,
                    'weight': g['weight'] * likelihood # Simplified weight update
                })
                
        # 2. Birth new Gaussians for unassociated measurements
        # TODO: Check if any measurement 'z' was not well-matched. 
        # If so, create a new Gaussian with high covariance.
        
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
        # TODO: If two Gaussians have means closer than self.merge_thresh, merge them.
        
        # 3. Cap maximum components
        if len(self.gaussians) > self.max_components:
            # Sort by weight and keep top N
            self.gaussians.sort(key=lambda g: g['weight'], reverse=True)
            self.gaussians = self.gaussians[:self.max_components]