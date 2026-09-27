import numpy as np

class MonocularLocator:
    def __init__(self, fx, fy, cx, cy, drone_radius, k1=0.0, k2=0.0):
        """
        Initialize with camera intrinsics and known neighbor drone radius.
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.r = drone_radius
        self.k1 = k1
        self.k2 = k2

    def correct_distortion(self, x, y):
        """
        Apply radial polynomial distortion correction.
        """
        # 1. Normalize coordinates
        xn = (x - self.cx) / self.fx
        yn = (y - self.cy) / self.fy
        
        # 2. Calculate radial distance squared
        r2 = xn**2 + yn**2
        
        # 3. Apply distortion factor
        factor = 1 + self.k1 * r2 + self.k2 * r2**2
        
        # 4. Correct and convert back to pixel coordinates
        x_corr = xn * factor * self.fx + self.cx
        y_corr = yn * factor * self.fy + self.cy
        
        return x_corr, y_corr

    def get_unit_vector(self, x, y):
        """
        Convert a corrected pixel coordinate to a 3D unit direction vector.
        """
        # Normalize to camera coordinate system (z=1 plane)
        x_cam = (x - self.cx) / self.fx
        y_cam = (y - self.cy) / self.fy
        z_cam = 1.0
        
        vec = np.array([x_cam, y_cam, z_cam])
        
        # Normalize to unit length
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            raise ValueError("Vector magnitude too small")
            
        return vec / norm

    def estimate_distance_and_bearing(self, x1, y1, x2, y2):
        """
        Main function implementing Eq. (3) and Eq. (4) from the paper.
        x1, y1: Left edge of bounding box
        x2, y2: Right edge of bounding box
        """
        # 1. Correct Distortion for both edges
        x1_c, y1_c = self.correct_distortion(x1, y1)
        x2_c, y2_c = self.correct_distortion(x2, y2)
        
        # 2. Get 3D Unit Direction Vectors (l1 and l2)
        l1 = self.get_unit_vector(x1_c, y1_c)
        l2 = self.get_unit_vector(x2_c, y2_c)
        
        # 3. Calculate Angle between vectors (Eq. 3)
        # cos(l) = l1 . l2 (since they are unit vectors)
        dot_product = np.dot(l1, l2)
        
        # Clamp to avoid numerical errors in acos
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # 4. Calculate Distance (Eq. 4)
        # d = r / sqrt((1 - cos(l)) / 2)
        denominator = np.sqrt((1 - dot_product) / 2)
        
        if denominator < 1e-6:
            raise ValueError("Target too far or bounding box too small")
            
        distance_d = self.r / denominator
        
        # 5. Calculate Bearing (Beta)
        # Beta is the average of l1 and l2
        beta_vec = (l1 + l2) / 2
        beta_vec = beta_vec / np.linalg.norm(beta_vec)
        
        # Bearing angle relative to camera z-axis (forward)
        bearing_angle = np.arctan2(beta_vec[0], beta_vec[2])
        
        return distance_d, bearing_angle

# --- Example Usage for Study ---
if __name__ == "__main__":
    # Simulated Camera Intrinsics
    fx, fy = 600, 600
    cx, cy = 320, 240
    drone_radius = 0.15 # 15cm radius
    
    locator = MonocularLocator(fx, fy, cx, cy, drone_radius)
    
    # Simulated Bounding Box (Left edge: 300,240; Right edge: 340,240)
    x1, y1 = 300, 240
    x2, y2 = 340, 240
    
    try:
        dist, bear = locator.estimate_distance_and_bearing(x1, y1, x2, y2)
        print(f"Estimated Distance: {dist:.2f} meters")
        print(f"Bearing Angle: {np.degrees(bear):.2f} degrees")
    except Exception as e:
        print(f"Error: {e}")