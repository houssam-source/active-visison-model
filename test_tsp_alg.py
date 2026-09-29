import unittest

import numpy as np

from GPNeighborBelief import UncertaintyQuantifier
from tsp_alg import TSPHeadingPlanner


class TSPHeadingPlannerTests(unittest.TestCase):
    def test_import_and_select_using_uncertainty_quantifier_grid(self):
        quantifier = UncertaintyQuantifier(
            x_range=(-3, 3),
            y_range=(-3, 3),
            resolution=1.0,
            max_fov_range=3.0,
        )
        uncertainty = np.zeros(quantifier.grid_shape)
        target_cell = (quantifier.X_grid == 0) & (quantifier.Y_grid == 2)
        uncertainty[target_cell] = 5.0
        planner = TSPHeadingPlanner(num_candidates=3, max_fov_range=quantifier.max_fov_range)

        heading, reward = planner.select_best_heading(
            uncertainty,
            quantifier.X_grid,
            quantifier.Y_grid,
            current_heading=0.0,
        )

        self.assertAlmostEqual(heading, np.pi / 2)
        self.assertEqual(reward, 5.0)

    def test_ties_keep_current_heading(self):
        quantifier = UncertaintyQuantifier(x_range=(-2, 2), y_range=(-2, 2), resolution=1.0)
        planner = TSPHeadingPlanner(num_candidates=5)

        heading, reward = planner.select_best_heading(
            np.zeros(quantifier.grid_shape),
            quantifier.X_grid,
            quantifier.Y_grid,
            current_heading=0.4,
        )

        self.assertAlmostEqual(heading, 0.4)
        self.assertEqual(reward, 0.0)

    def test_rejects_grid_shape_mismatch(self):
        planner = TSPHeadingPlanner()

        with self.assertRaisesRegex(ValueError, "matching shapes"):
            planner.select_best_heading(np.ones((2, 2)), np.ones((2, 2)), np.ones((3, 2)), 0.0)


if __name__ == "__main__":
    unittest.main()