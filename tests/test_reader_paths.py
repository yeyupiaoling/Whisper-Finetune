"""Offline regression tests for dataset manifest path expansion."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.reader import (
    CustomDataset, DistillWhisperDataset, MoonshineDataset,
    load_data_list, resolve_data_list_paths,
)


class ReaderPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_manifest(self, name, rows):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_single_file_and_explicit_list_order(self):
        first = self.write_manifest("a.json", [])
        second = self.write_manifest("b.jsonl", [])
        self.assertEqual(resolve_data_list_paths(str(first)), [str(first)])
        self.assertEqual(resolve_data_list_paths(first), [str(first)])
        for paths in ([second, first], (second, first)):
            self.assertEqual(resolve_data_list_paths(paths), [str(second), str(first)])

    def test_directory_sorted_and_non_recursive(self):
        second = self.write_manifest("data/b.jsonl", [])
        first = self.write_manifest("data/a.json", [])
        self.write_manifest("data/notes.txt", [])
        self.write_manifest("data/nested/hidden.json", [])
        (self.root / "data/subdir.json").mkdir()
        expected = [str(first), str(second)]
        self.assertEqual(resolve_data_list_paths(self.root / "data"), expected)
        self.assertEqual(resolve_data_list_paths(str(self.root / "data")), expected)
        extra = self.write_manifest("extra.json", [])
        self.assertEqual(resolve_data_list_paths([extra, first.parent]), [str(extra), *expected])

    def test_empty_directory_and_invalid_inputs(self):
        self.write_manifest("notes.txt", [])
        with self.assertRaisesRegex(ValueError, "JSON/JSONL"):
            resolve_data_list_paths(self.root)
        for invalid in (None, 123, [123]):
            with self.subTest(invalid=invalid), self.assertRaises(TypeError):
                resolve_data_list_paths(invalid)
        with self.assertRaises(FileNotFoundError):
            load_data_list(self.root / "missing.json", 0.5, 30, 1, 200)

    def test_load_directory_preserves_filters_and_dataset_support(self):
        first = {"duration": 1, "sentence": "你好"}
        second = {"duration": 2, "sentences": [{"text": "世界"}]}
        self.write_manifest("a.json", [first, {"duration": 0.1, "sentence": "short"}])
        self.write_manifest("b.jsonl", [second, {"duration": 31, "sentence": "long"},
                                         {"duration": 1, "sentence": ""},
                                         {"duration": 1, "sentence": "x" * 201}])
        expected = [first, second]
        self.assertEqual(load_data_list(self.root, 0.5, 30, 1, 200), expected)
        # Exercise the actual shared loading methods without a model or audio files.
        for dataset_class in (CustomDataset, DistillWhisperDataset, MoonshineDataset):
            with self.subTest(dataset=dataset_class.__name__):
                dataset = SimpleNamespace(data_list_path=self.root, min_duration=0.5,
                                          max_duration=30, min_sentence=1, max_sentence=200)
                dataset_class._load_data_list(dataset)
                self.assertEqual(dataset.data_list, expected)


if __name__ == "__main__":
    unittest.main()