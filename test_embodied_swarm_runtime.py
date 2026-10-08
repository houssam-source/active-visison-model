import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from swarmmanager import EmbodiedSwarmRuntime


class EmbodiedSwarmRuntimeTests(unittest.TestCase):
    def test_runtime_uses_explicit_model_path(self):
        with TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "runtime_model.pt"
            model_path.write_bytes(b"model")

            runtime = EmbodiedSwarmRuntime(
                n_drones=2,
                detector_model_path=str(model_path),
                skip_simulation=True,
            )

            self.assertEqual(runtime.model_path, model_path.resolve())
            self.assertEqual(runtime.n_drones, 2)

    def test_runtime_summary_lists_active_agents(self):
        runtime = EmbodiedSwarmRuntime(n_drones=3, skip_simulation=True)
        summary = runtime.summary()

        self.assertEqual(summary["n_drones"], 3)
        self.assertEqual(summary["active_agents"], 3)
        self.assertIn("mode", summary)


if __name__ == "__main__":
    unittest.main()
