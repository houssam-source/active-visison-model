import numpy as np


class DistortionCorrector:
    """Standalone distortion-correction component for a monocular camera."""

    def __init__(self, fx, fy, cx, cy, k1=0.0, k2=0.0):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def correct(self, x, y):
        """Apply radial polynomial distortion correction to one pixel."""
        xn = (x - self.cx) / self.fx
        yn = (y - self.cy) / self.fy

        r2 = xn**2 + yn**2
        factor = 1 + self.k1 * r2 + self.k2 * r2**2

        x_corr = xn * factor * self.fx + self.cx
        y_corr = yn * factor * self.fy + self.cy
        return x_corr, y_corr

    def get_unit_vector(self, x, y):
        """Convert a corrected pixel coordinate to a unit camera vector."""
        x_cam = (x - self.cx) / self.fx
        y_cam = (y - self.cy) / self.fy
        vec = np.array([x_cam, y_cam, 1.0], dtype=float)

        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            raise ValueError("Vector magnitude too small")

        return vec / norm


class MonocularLocator(DistortionCorrector):
    """Compatibility wrapper preserving the original project interface."""

    def __init__(self, fx, fy, cx, cy, drone_radius, k1=0.0, k2=0.0):
        super().__init__(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2)
        self.r = float(drone_radius)

    def correct_distortion(self, x, y):
        return self.correct(x, y)

    def estimate_distance_and_bearing(self, x1, y1, x2, y2):
        x1_c, y1_c = self.correct_distortion(x1, y1)
        x2_c, y2_c = self.correct_distortion(x2, y2)

        l1 = self.get_unit_vector(x1_c, y1_c)
        l2 = self.get_unit_vector(x2_c, y2_c)

        dot_product = np.dot(l1, l2)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        denominator = np.sqrt((1 - dot_product) / 2)
        if denominator < 1e-6:
            raise ValueError("Target too far or bounding box too small")

        distance_d = self.r / denominator

        beta_vec = (l1 + l2) / 2
        beta_vec = beta_vec / np.linalg.norm(beta_vec)
        bearing_angle = np.arctan2(beta_vec[0], beta_vec[2])

        return distance_d, bearing_angle