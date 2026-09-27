import numpy as np

from distortion_correction import DistortionCorrector


def test_center_pixel_is_unchanged_without_distortion():
    corrector = DistortionCorrector(fx=600, fy=600, cx=320, cy=240)

    x_corr, y_corr = corrector.correct(320, 240)

    assert x_corr == 320
    assert y_corr == 240


def test_principal_axis_unit_vector_is_forward():
    corrector = DistortionCorrector(fx=600, fy=600, cx=320, cy=240)

    vec = corrector.get_unit_vector(320, 240)

    np.testing.assert_allclose(vec, np.array([0.0, 0.0, 1.0]))
