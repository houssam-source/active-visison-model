import unittest

import numpy as np

from distortion_correction import DistortionCorrector


class DistortionCorrectorTests(unittest.TestCase):
    def test_center_pixel_is_unchanged_without_distortion(self):
        corrector = DistortionCorrector(fx=600, fy=600, cx=320, cy=240)

        x_corr, y_corr = corrector.correct(320, 240)

        self.assertEqual(x_corr, 320)
        self.assertEqual(y_corr, 240)

    def test_principal_axis_unit_vector_is_forward(self):
        corrector = DistortionCorrector(fx=600, fy=600, cx=320, cy=240)

        vec = corrector.get_unit_vector(320, 240)

        np.testing.assert_allclose(vec, np.array([0.0, 0.0, 1.0]))

    def test_invalid_focal_length_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Focal lengths must be positive"):
            DistortionCorrector(fx=0, fy=600, cx=320, cy=240)
