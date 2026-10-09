import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jag.monitoring import MonitoringSession, public_config, read_swanlab_api_key


class MonitoringTests(unittest.TestCase):
    def test_logging_is_off_by_default(self):
        with TemporaryDirectory() as directory, patch.object(MonitoringSession, "_connect") as connect:
            session = MonitoringSession(directory, {})
            self.assertFalse(session.enabled)
            connect.assert_not_called()

    def test_credential_environment_precedence_and_file_fallback(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "credential"
            path.write_text(" example-file-value \n")
            with patch.dict("os.environ", {"SWANLAB_API_KEY": " example-env-value ",
                                           "SWANLAB_API_KEY_FILE": str(path)}, clear=True):
                self.assertEqual(read_swanlab_api_key(), "example-env-value")
            with patch.dict("os.environ", {"SWANLAB_API_KEY_FILE": str(path)}, clear=True):
                self.assertEqual(read_swanlab_api_key(), "example-file-value")
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "SWANLAB_API_KEY"):
                    read_swanlab_api_key()

    def test_config_omits_nested_credentials(self):
        self.assertEqual(public_config({"model": "qwen", "nested": {"api_key": "secret", "epochs": 2}}),
                         {"model": "qwen", "nested": {"epochs": 2}})

    def test_eta_includes_remaining_joint_stage(self):
        with TemporaryDirectory() as directory:
            calibration = Path(directory) / "prior.json"
            calibration.write_text(json.dumps({"stages": {
                "head": {"seconds_per_step": 2, "effective_batch_size": 96},
                "joint": {"seconds_per_step": 10, "effective_batch_size": 96},
            }}))
            session = MonitoringSession(directory, {}, enabled=False, calibration_path=calibration)
            callback = session.callbacks("head", future_joint_steps=200)[0]
            callback.effective_batch = 96
            progress = callback._progress(SimpleNamespace(global_step=3, max_steps=100, epoch=0.1))
            self.assertEqual(progress["eta_seconds"], 97 * 2 + 200 * 10)
            self.assertEqual(progress["estimate_basis"], "preflight_calibration")
            callback.durations.extend([3, 100, 4])
            progress = callback._progress(SimpleNamespace(global_step=4, max_steps=100, epoch=0.1))
            self.assertEqual(progress["eta_seconds"], 96 * 4 + 200 * 10)

    def test_unknown_joint_rate_does_not_claim_total_eta(self):
        with TemporaryDirectory() as directory:
            session = MonitoringSession(directory, {}, enabled=False)
            callback = session.callbacks("head", future_joint_steps=200)[0]
            callback.effective_batch = 96
            callback.durations.append(2)
            progress = callback._progress(SimpleNamespace(global_step=1, max_steps=100, epoch=0.1))
            self.assertEqual(progress["stage_remaining_seconds"], 198)
            self.assertIsNone(progress["eta_seconds"])

    def test_nonzero_rank_never_connects_or_writes(self):
        with TemporaryDirectory() as directory, patch.dict("os.environ", {"RANK": "1"}):
            target = Path(directory) / "rank1"
            session = MonitoringSession(target, {}, enabled=True)
            self.assertEqual(session.callbacks("joint"), [])
            session.finish()
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
