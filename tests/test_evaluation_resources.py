import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jag.resources import configure_cuda_memory


class EvaluationResourceTests(unittest.TestCase):
    def test_no_limit_does_not_touch_cuda(self):
        with patch("torch.cuda.is_available", side_effect=AssertionError("CUDA queried")):
            self.assertFalse(configure_cuda_memory("cpu")["allocator_limit_applied"])

    def test_invalid_limits_rejected(self):
        for kwargs in ({"memory_gib": -1}, {"memory_gib": float("nan")},
                       {"memory_fraction": 1.1}, {"memory_fraction": 0},
                       {"memory_gib": 8, "memory_fraction": .2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                configure_cuda_memory("cuda:0", **kwargs)
        with self.assertRaises(ValueError):
            configure_cuda_memory("cpu", memory_gib=8)

    def test_cap_precedes_loading_and_preserves_other_process_memory(self):
        gib = 1024**3
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=40*gib)), \
             patch("torch.cuda.mem_get_info", return_value=(20*gib, 40*gib)), \
             patch("torch.cuda.set_per_process_memory_fraction") as limit:
            result = configure_cuda_memory("cuda:0", memory_gib=8)
            limit.assert_called_once_with(.2, 0)
            self.assertEqual(result["allocator_limit_bytes"], 8*gib)

    def test_low_headroom_refuses_evaluation(self):
        gib = 1024**3
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=40*gib)), \
             patch("torch.cuda.mem_get_info", return_value=(9*gib, 40*gib)), \
             patch("torch.cuda.set_per_process_memory_fraction") as limit:
            with self.assertRaisesRegex(RuntimeError, "headroom"):
                configure_cuda_memory("cuda:0", memory_gib=8)
            limit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
