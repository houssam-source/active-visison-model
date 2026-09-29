import unittest

import numpy as np

from GMHPHDtracker import GMPHDTracker


class GMPHDTrackerTests(unittest.TestCase):
    def make_tracker(self, merge_thresh=0.0):
        return GMPHDTracker(
            dt=1.0,
            process_noise_std=0.1,
            dist_noise_std=0.5,
            bear_noise_std=0.1,
            prune_thresh=0.0,
            merge_thresh=merge_thresh,
        )

    def test_measurement_updates_each_plausible_component(self):
        tracker = self.make_tracker()
        for x_position in (10.0, 10.2):
            tracker.gaussians.append({
                "mean": np.array([x_position, 0.0, 0.0, 0.0]),
                "cov": np.eye(4),
                "weight": 1.0,
            })

        tracker.update([[10.0, 0.0]])

        self.assertEqual(len(tracker.gaussians), 2)

    def test_merge_includes_between_component_variance(self):
        tracker = self.make_tracker(merge_thresh=3.0)
        tracker.gaussians = [
            {"mean": np.array([0.0, 0.0, 0.0, 0.0]), "cov": np.eye(4), "weight": 1.0},
            {"mean": np.array([2.0, 0.0, 0.0, 0.0]), "cov": np.eye(4), "weight": 1.0},
        ]

        tracker._prune_and_merge()

        self.assertEqual(len(tracker.gaussians), 1)
        np.testing.assert_allclose(tracker.gaussians[0]["mean"], [1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(tracker.gaussians[0]["cov"][0, 0], 2.0)

    def test_update_keeps_covariance_finite_symmetric_and_psd(self):
        tracker = self.make_tracker()
        tracker.gaussians = [{
            "mean": np.array([10.0, 0.0, 0.0, 0.0]),
            "cov": np.eye(4),
            "weight": 1.0,
        }]

        tracker.update([[10.1, 0.02]])

        covariance = tracker.gaussians[0]["cov"]
        self.assertTrue(np.isfinite(covariance).all())
        np.testing.assert_allclose(covariance, covariance.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-10)

    def test_predict_rejects_invalid_ego_velocity(self):
        tracker = self.make_tracker()

        with self.assertRaisesRegex(ValueError, "two finite values"):
            tracker.predict([1.0, np.nan])

    def test_update_rejects_negative_measurement_range(self):
        tracker = self.make_tracker()

        with self.assertRaisesRegex(ValueError, "positive ranges"):
            tracker.update([[-1.0, 0.0]])

    def test_update_rejects_zero_measurement_range(self):
        tracker = self.make_tracker()

        with self.assertRaisesRegex(ValueError, "positive ranges"):
            tracker.update([[0.0, 0.0]])

    def test_origin_component_does_not_claim_measurement(self):
        tracker = self.make_tracker()
        tracker.gaussians = [{
            "mean": np.zeros(4),
            "cov": np.eye(4),
            "weight": 1.0,
        }]

        tracker.update([[0.1, 0.0]])

        self.assertEqual(len(tracker.gaussians), 1)
        np.testing.assert_allclose(tracker.gaussians[0]["mean"], [0.1, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()