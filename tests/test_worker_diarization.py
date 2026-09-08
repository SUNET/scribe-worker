"""Worker contract test using a fake pipeline, without downloading models."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from test_diarization import Annotation


class WorkerResultTests(unittest.TestCase):
    def test_words_split_transcript_without_changing_raw_intervals(self):
        # Settings parses the worker CLI on import, independently of unittest.
        with patch("sys.argv", ["scribe-worker"]):
            from utils.whisper import WhisperAudioTranscriber

        annotation = Annotation([(0.1, 2.0, "SPEAKER_00"),
                                 (1.8, 3.0, "SPEAKER_01"),
                                 (10.0, 11.0, "SPEAKER_01")])
        annotation.labels = lambda: ["SPEAKER_00", "SPEAKER_01"]
        pipeline = Mock(return_value=SimpleNamespace(speaker_diarization=annotation))
        worker = WhisperAudioTranscriber.__new__(WhisperAudioTranscriber)
        for name, value in {
            "logger": Mock(), "speakers": 2,
            "diarization_pipeline": pipeline,
            "chunks": [{"start": 0.0, "end": 3.0, "text": "Hello there.",
                        "avg_score": 0.9, "words": [
                            {"t": "Hello", "s": 0.1, "e": 1.5, "c": 0.9},
                            {"t": "there.", "s": 2.1, "e": 2.9, "c": 0.8},
                        ]}],
            "audio_data": None, "audio_path": "unused.wav",
            "full_transcription": "Hello there.",
        }.items():
            setattr(worker, "_WhisperAudioTranscriber__" + name, value)

        result = worker.diarization()
        self.assertEqual(result["full_transcription"], "Hello there.")
        self.assertEqual(result["speaker_count"], 2)
        self.assertEqual(len(result["segments"]), 2)
        segment = result["segments"][0]
        self.assertEqual(segment["speaker"], "Speaker_00")
        self.assertEqual(segment["active_speakers"], ["Speaker_00"])
        self.assertEqual(segment["text"], "Hello")
        self.assertEqual(segment["avg_score"], 0.9)
        self.assertEqual(result["segments"][1]["speaker"], "Speaker_01")
        self.assertEqual(" ".join(s["text"] for s in result["segments"]), "Hello there.")
        self.assertEqual(len(result["diarization_segments"]), 3)
        self.assertEqual(result["diarization_segments"][-1],
                         {"start": 10.0, "end": 11.0, "speaker": "Speaker_01"})
        pipeline.assert_called_once()

    def test_normalization_keeps_words_attached_when_long_text_is_split(self):
        with patch("sys.argv", ["scribe-worker"]):
            from utils.whisper import WhisperAudioTranscriber, settings
        from utils.speaker_alignment import align_chunks

        worker = WhisperAudioTranscriber.__new__(WhisperAudioTranscriber)
        words = [{"text": text, "start": index + 0.1, "end": index + 0.9}
                 for index, text in enumerate(["One", "two", "three", "four"])]
        with patch.object(settings, "SEGMENT_SPLIT_LENGTH", 10):
            worker._WhisperAudioTranscriber__process_transcription([
                {"start": 0.0, "end": 4.0, "text": "One two three four", "words": words}
            ])
        chunks = worker._WhisperAudioTranscriber__chunks
        self.assertEqual(len(chunks), 2)
        self.assertEqual([w["t"] for c in chunks for w in c["words"]],
                         ["One", "two", "three", "four"])
        result = align_chunks(chunks, [
            {"start": 0, "end": 1, "speaker": "A"},
            {"start": 1, "end": 4, "speaker": "B"},
        ])
        self.assertEqual([s["speaker"] for s in result], ["A", "B", "B"])
        self.assertEqual(" ".join(s["text"] for s in result), "One two three four")
        self.assertEqual(worker._WhisperAudioTranscriber__full_transcription,
                         "One two three four")
