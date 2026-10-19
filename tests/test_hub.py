"""Check local loading stays offline and Hub IDs resolve automatically."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jag.hub import resolve_model_directory


class HubTests(unittest.TestCase):
    def test_local_model_does_not_access_the_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("huggingface_hub.snapshot_download") as download:
                self.assertEqual(resolve_model_directory(directory), Path(directory))
                download.assert_not_called()

    def test_model_id_downloads_into_the_hub_cache(self):
        with patch("huggingface_hub.snapshot_download", return_value="/cache/jag") as download:
            self.assertEqual(resolve_model_directory("xyzzzh/Jag"), Path("/cache/jag"))
            self.assertEqual(download.call_args.kwargs["repo_id"], "xyzzzh/Jag")

    def test_missing_explicit_local_directory_does_not_download(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("huggingface_hub.snapshot_download") as download:
                with self.assertRaises(FileNotFoundError):
                    resolve_model_directory(Path(directory) / "missing")
                download.assert_not_called()

    def test_file_is_not_a_model_directory(self):
        with tempfile.NamedTemporaryFile() as model_file:
            with self.assertRaises(FileNotFoundError):
                resolve_model_directory(model_file.name)


if __name__ == "__main__":
    unittest.main()
