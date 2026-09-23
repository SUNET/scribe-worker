import json
import unittest
from types import SimpleNamespace

from utils.diarization import serialize_diarization


class Annotation:
    def __init__(self, turns):
        self.turns = turns

    def itertracks(self, yield_label=False):
        assert yield_label
        for start, end, speaker in self.turns:
            yield SimpleNamespace(start=start, end=end), "track", speaker


class SerializationTests(unittest.TestCase):
    def test_preserves_overlap_precision_and_untranscribed_turns(self):
        annotation = Annotation([
            (0.123456789, 2.5, "SPEAKER_00"),
            (1.7, 3.0, "SPEAKER_01"),
            (20.0, 20.1, "SPEAKER_00"),
        ])
        result = serialize_diarization(annotation)
        self.assertEqual(json.loads(json.dumps(result)), [
            {"start": 0.123456789, "end": 2.5, "speaker": "Speaker_00"},
            {"start": 1.7, "end": 3.0, "speaker": "Speaker_01"},
            {"start": 20.0, "end": 20.1, "speaker": "Speaker_00"},
        ])

    def test_empty_annotation(self):
        self.assertEqual(serialize_diarization(Annotation([])), [])

    def test_preserves_nonstandard_labels(self):
        result = serialize_diarization(Annotation([(0, 1, "Alice")]))
        self.assertEqual(result[0]["speaker"], "Alice")


if __name__ == "__main__":
    unittest.main()
