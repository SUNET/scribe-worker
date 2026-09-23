import copy
import unittest

from utils.speaker_alignment import activity_spans, align_chunks


def turn(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def chunk(text="One two three", words=None):
    return {"start": 0.0, "end": 3.0, "text": text, "words": words or [
        {"t": "One", "s": 0.1, "e": 0.9},
        {"t": "two", "s": 1.1, "e": 1.9},
        {"t": "three", "s": 2.1, "e": 2.9},
    ]}


class AlignmentTests(unittest.TestCase):
    def test_brief_midpoint_interjection_does_not_label_whole_chunk(self):
        data = chunk()
        result = align_chunks([data], [turn(0, 1, "A"), turn(1, 2, "B"), turn(2, 3, "A")])
        self.assertEqual([s["speaker"] for s in result], ["A", "B", "A"])
        self.assertEqual([s["text"] for s in result], ["One", "two", "three"])
        self.assertEqual([s["active_speakers"] for s in result], [["A"], ["B"], ["A"]])
        self.assertEqual([(s["start"], s["end"]) for s in result], [(0, 1), (1, 2), (2, 3)])

    def test_real_overlap_is_explicit_and_has_no_arbitrary_primary_speaker(self):
        result = align_chunks([chunk()], [turn(0, 3, "A"), turn(1, 2, "B")])
        self.assertEqual([s["speaker"] for s in result], ["A", "Unknown", "A"])
        self.assertEqual(result[1]["active_speakers"], ["A", "B"])

    def test_silence_is_unknown_not_speaker_zero(self):
        result = align_chunks([chunk()], [])
        self.assertEqual(result[0]["speaker"], "Unknown")
        self.assertEqual(result[0]["active_speakers"], [])
        self.assertEqual(result[0]["text"], "One two three")

    def test_midpoint_silence_does_not_override_words(self):
        result = align_chunks([chunk()], [turn(0, 1.4, "A"), turn(1.6, 3, "A")])
        self.assertEqual([s["speaker"] for s in result], ["A"])

    def test_equal_sequential_overlap_is_ambiguous_not_simultaneous(self):
        data = chunk("word", [{"t": "word", "s": 1, "e": 2}])
        result = align_chunks([data], [turn(0, 1.5, "A"), turn(1.5, 3, "B")])
        self.assertEqual(result[0]["speaker"], "Unknown")
        self.assertEqual(result[0]["active_speakers"], [])

    def test_missing_words_preserve_text_without_inventing_alignment(self):
        data = {"start": 0, "end": 3, "text": "Important omitted words!"}
        result = align_chunks([data], [turn(0, 1, "A"), turn(1, 3, "B")])
        self.assertEqual(result[0]["text"], data["text"])
        self.assertEqual(result[0]["speaker"], "Unknown")
        self.assertEqual(result[0]["active_speakers"], [])

    def test_missing_words_with_one_speaker(self):
        result = align_chunks([{"start": 0, "end": 3, "text": "Text"}], [turn(0, 3, "A")])
        self.assertEqual(result[0]["speaker"], "A")

    def test_mismatched_word_payload_does_not_drop_text(self):
        data = chunk("One EXTRA two three!")
        result = align_chunks([data], [turn(0, 1, "A"), turn(1, 3, "B")])
        self.assertEqual(result[0]["text"], data["text"])
        self.assertEqual(result[0]["speaker"], "Unknown")

    def test_punctuation_and_confidence_survive_split(self):
        data = chunk("Hej,  ja!", [{"t": "Hej,", "s": 0.2, "e": 1, "c": 0.9},
                                  {"t": "ja!", "s": 2, "e": 2.5, "c": 0.7}])
        original = copy.deepcopy(data)
        turns = [turn(0, 1.5, "A"), turn(1.5, 3, "B")]
        result = align_chunks([data], turns)
        self.assertEqual([s["text"] for s in result], ["Hej,", "ja!"])
        self.assertEqual([s["avg_score"] for s in result], [0.9, 0.7])
        self.assertEqual(data, original)

    def test_duplicate_tracks_do_not_end_speaker_activity_early(self):
        spans = activity_spans([turn(0, 2, "A"), turn(1, 3, "A"), turn(3, 4, "B")])
        self.assertEqual(spans, [(0, 1, ("A",)), (1, 2, ("A",)),
                                 (2, 3, ("A",)), (3, 4, ("B",))])

    def test_zero_duration_word_is_preserved_without_guessing(self):
        data = chunk("word", [{"t": "word", "s": 1, "e": 1}])
        result = align_chunks([data], [turn(0, 1, "A"), turn(1, 3, "B")])
        self.assertEqual(result[0]["text"], "word")
        self.assertEqual(result[0]["speaker"], "Unknown")

    def test_rounding_at_chunk_boundaries_does_not_disable_alignment(self):
        data = chunk()
        data["start"] = 0.1004
        result = align_chunks([data], [turn(0, 1, "A"), turn(1, 3, "B")])
        self.assertEqual([s["speaker"] for s in result], ["A", "B"])
