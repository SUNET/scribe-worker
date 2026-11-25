import json
import unittest
from unittest.mock import patch

# Settings parses the worker CLI on import, independently of unittest.
with patch("sys.argv", ["scribe-worker"]):
    from utils.diarization import serialize_diarization


class SerializationTests(unittest.TestCase):
    def test_preserves_overlap_precision_and_untranscribed_turns(self):
        segments = [
            (0.123456789, 2.5, "Speaker_00"),
            (1.7, 3.0, "Speaker_01"),
            (20.0, 20.1, "Speaker_00"),
        ]
        result = serialize_diarization(segments)
        self.assertEqual(
            json.loads(json.dumps(result)),
            [
                {"start": 0.123456789, "end": 2.5, "speaker": "Speaker_00"},
                {"start": 1.7, "end": 3.0, "speaker": "Speaker_01"},
                {"start": 20.0, "end": 20.1, "speaker": "Speaker_00"},
            ],
        )

    def test_empty_segments(self):
        self.assertEqual(serialize_diarization([]), [])

    def test_converts_numeric_bounds_to_float(self):
        result = serialize_diarization([(0, 1, "Speaker_00")])
        self.assertIsInstance(result[0]["start"], float)
        self.assertIsInstance(result[0]["end"], float)
        self.assertEqual(result[0], {"start": 0.0, "end": 1.0, "speaker": "Speaker_00"})


if __name__ == "__main__":
    unittest.main()
