import numpy as np

from swarmmanager import SwarmActiveVisionEnv


class _FakeCamera:
    def render(self, rgb=True, depth=True):
        raise RuntimeError("headless render failed")


def test_render_camera_safe_produces_non_black_fallback_frame():
    env = object.__new__(SwarmActiveVisionEnv)
    env.camera_resolution = (640, 480)
    env.last_render_error = None

    frame = {"heading": 0.75, "camera": _FakeCamera()}
    rgb, depth, _, _ = SwarmActiveVisionEnv._render_camera_safe(env, frame)

    assert rgb.shape == (480, 640, 3)
    assert depth.shape == (480, 640)
    assert (rgb > 20).mean() > 0.01
    assert env.last_render_error is not None
