import numpy as np
import numpy.testing as npt

from GPNeighborBelief import UncertaintyQuantifier

# ---------------------------------------------------------
# MOCK GP (Matches your GPNeighborBelief interface)
# ---------------------------------------------------------
class MockGP:
    def __init__(self, sigma_value, has_history=True):
        self.sigma = sigma_value
        self.has_history = has_history

    def prune_old_data(self, t_now):
        pass  # No-op for testing

    def predict(self, x, y, t_now):
        # Simulates: prior sigma if no history, else fixed sigma
        sigma = self.sigma if self.has_history else 1.0
        return 0.0, sigma

# ---------------------------------------------------------
# UNIT TESTS
# ---------------------------------------------------------
def test_init_and_grid():
    """Normal: Verify grid creation, shape, and coordinate ranges."""
    q = UncertaintyQuantifier(x_range=(-5, 5), y_range=(-3, 3), resolution=1.0)
    assert q.X_grid.shape == (6, 10), "Grid shape mismatch"
    assert np.allclose(q.x, np.arange(-5, 5, 1.0))
    assert np.allclose(q.y, np.arange(-3, 3, 1.0))
    print("✅ PASS: Grid initialization & coordinates")

def test_fov_mask_heading_zero():
    """Normal: 90° cone facing +X should mask correct quadrant."""
    q = UncertaintyQuantifier(x_range=(-4, 4), y_range=(-4, 4), resolution=1.0)
    mask = q._generate_fov_mask(0.0)

    # (2, 0) should be INSIDE (0°)
    assert mask[(q.Y_grid == 0) & (q.X_grid == 2)].item()
    # (0, 2) should be OUTSIDE (90° > 45°)
    assert not mask[(q.Y_grid == 2) & (q.X_grid == 0)].item()
    # (2, 1) should be INSIDE (~26.5° < 45°)
    assert mask[(q.X_grid == 2) & (q.Y_grid == 1)].item()
    print("✅ PASS: FOV mask geometry (heading=0)")

def test_fov_mask_boundary_conditions():
    """Edge: Exact 45° boundary & max range limit."""
    q = UncertaintyQuantifier(x_range=(-5, 5), y_range=(-5, 5), resolution=0.5, max_fov_range=3.0)
    mask = q._generate_fov_mask(np.pi / 4)  # Heading 45°

    # Point exactly at 45° relative should be INSIDE (<= condition)
    rel_45 = (q.X_grid == 0) & (q.Y_grid == 2)
    assert mask[rel_45].item()

    # Point outside max range should be OUTSIDE regardless of angle
    out_range = (q.X_grid == 4) & (q.Y_grid == 4)
    assert out_range.any()
    assert not mask[out_range].item()
    print("✅ PASS: FOV boundary conditions (angle & range)")

def test_unc_normal_single_gp():
    """Normal: Single GP reduces U only inside FOV."""
    q = UncertaintyQuantifier(resolution=2.0, max_fov_range=5.0)
    gp = MockGP(sigma_value=0.1, has_history=True)
    U = q.compute_UNC_matrix([gp], uav_heading=0.0, t_now=5.0)

    # Inside FOV: U should match sigma
    in_fov = q._generate_fov_mask(0.0)
    npt.assert_allclose(U[in_fov], 0.1, atol=1e-5)
    # Outside FOV: U should remain 1.0
    npt.assert_allclose(U[~in_fov], 1.0, atol=1e-5)
    print("✅ PASS: UNC normal case (single GP, FOV min logic)")

def test_unc_multiple_gps_overlap():
    """Normal: Multiple GPs → U takes minimum sigma in overlap."""
    q = UncertaintyQuantifier(resolution=2.0, max_fov_range=5.0)
    gp1 = MockGP(sigma_value=0.3, has_history=True)
    gp2 = MockGP(sigma_value=0.1, has_history=True)

    U = q.compute_UNC_matrix([gp1, gp2], uav_heading=0.0, t_now=5.0)
    in_fov = q._generate_fov_mask(0.0)

    # Should pick the lower sigma (0.1)
    npt.assert_allclose(U[in_fov], 0.1, atol=1e-5)
    print("✅ PASS: UNC multi-GP overlap (min logic)")

def test_unc_empty_gps_list():
    """Edge: Empty GP list → U remains 1.0 everywhere."""
    q = UncertaintyQuantifier(resolution=1.0)
    U = q.compute_UNC_matrix([], uav_heading=0.0, t_now=0.0)
    assert np.all(U == 1.0), "Empty GP list should return max uncertainty"
    print("✅ PASS: UNC empty GP list (max uncertainty preserved)")

def test_unc_no_history_gp():
    """Edge: GP with no data → returns prior sigma_f, mask still applies."""
    q = UncertaintyQuantifier(resolution=1.0, max_fov_range=5.0)
    gp = MockGP(sigma_value=0.1, has_history=False)  # Returns 1.0 internally
    U = q.compute_UNC_matrix([gp], uav_heading=0.0, t_now=5.0)

    # Even with mask, sigma=1.0 → min(1.0, 1.0) = 1.0
    assert np.all(U == 1.0)
    print("✅ PASS: UNC no-history GP (prior fallback)")

def test_fov_heading_wraparound():
    """Edge: Heading near ±π should normalize correctly."""
    q = UncertaintyQuantifier(x_range=(-5, 5), y_range=(-5, 5), resolution=1.0)
    mask1 = q._generate_fov_mask(np.pi + 0.01)
    mask2 = q._generate_fov_mask(-np.pi + 0.01)
    # Should be identical due to normalization
    assert np.array_equal(mask1, mask2)
    print("✅ PASS: Heading wraparound normalization")

# ---------------------------------------------------------
# RUN ALL TESTS
# ---------------------------------------------------------
if __name__ == "__main__":
    test_init_and_grid()
    test_fov_mask_heading_zero()
    test_fov_mask_boundary_conditions()
    test_unc_normal_single_gp()
    test_unc_multiple_gps_overlap()
    test_unc_empty_gps_list()
    test_unc_no_history_gp()
    test_fov_heading_wraparound()
    print("\n🚀 ALL TESTS PASSED. UNC QUANTIFIER IS BULLET-PROOF.")
