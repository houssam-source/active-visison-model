import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from genesis_bridge import build_measurement_provider, resolve_detector_model_path


class UnifiedRuntimeContractTests(unittest.TestCase):
    def test_resolve_detector_model_path_prefers_explicit_path(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "custom_model.pt"
            model_path.write_bytes(b"model")

            self.assertEqual(
                resolve_detector_model_path(model_path),
                model_path.resolve(),
            )

    def test_resolve_detector_model_path_prefers_env_override(self):
        with TemporaryDirectory() as tmpdir:
            env_model = Path(tmpdir) / "env_model.pt"
            default_model = Path(tmpdir) / "default_model.pt"
            env_model.write_bytes(b"env")
            default_model.write_bytes(b"default")

            with mock.patch.dict(os.environ, {"YOLO_MODEL_PATH": str(env_model)}, clear=False):
                self.assertEqual(
                    resolve_detector_model_path(default_paths=[default_model]),
                    env_model.resolve(),
                )

    def test_resolve_detector_model_path_raises_for_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            resolve_detector_model_path(Path("missing_checkpoint.pt"))

    def test_build_measurement_provider_prefers_custom_provider(self):
        provider = lambda *args, **kwargs: []
        self.assertIs(build_measurement_provider(provider), provider)


if __name__ == "__main__":
    unittest.main()
