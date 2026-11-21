"""Worker contract test using a fake diarizer, without downloading models."""

import unittest
from unittest.mock import Mock, patch


class WorkerResultTests(unittest.TestCase):
    def test_additive_intervals_leave_existing_result_unchanged(self):
        # Settings parses the worker CLI on import, independently of unittest.
        with patch("sys.argv", ["scribe-worker"]):
            from utils.whisper import WhisperAudioTranscriber

        raw_segments = [
            (0.1, 2.0, "Speaker_00"),
            (1.8, 3.0, "Speaker_01"),
            (10.0, 11.0, "Speaker_01"),
        ]
        diarizer = Mock()
        diarizer.diarize.return_value = raw_segments

        worker = WhisperAudioTranscriber.__new__(WhisperAudioTranscriber)
        for name, value in {
            "logger": Mock(),
            "speakers": 2,
            "diarizer": diarizer,
            "chunks": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "Hello there.",
                    "avg_score": 0.9,
                }
            ],
            "audio_data": None,
            "audio_path": "unused.wav",
            "full_transcription": "Hello there.",
        }.items():
            setattr(worker, "_WhisperAudioTranscriber__" + name, value)

        result = worker.diarization()
        self.assertEqual(result["full_transcription"], "Hello there.")
        self.assertEqual(result["speaker_count"], 2)
        self.assertEqual(len(result["segments"]), 1)
        segment = result["segments"][0]
        self.assertEqual(segment["speaker"], "Speaker_00")
        self.assertEqual(set(segment["active_speakers"]), {"Speaker_00", "Speaker_01"})
        self.assertEqual(segment["text"], "Hello there.")
        self.assertEqual(segment["avg_score"], 0.9)
        self.assertEqual(
            (segment["start"], segment["end"], segment["duration"]),
            (0.0, 3.0, 3.0),
        )
        self.assertEqual(
            result["diarization_segments"],
            [
                {"start": 0.1, "end": 2.0, "speaker": "Speaker_00"},
                {"start": 1.8, "end": 3.0, "speaker": "Speaker_01"},
                {"start": 10.0, "end": 11.0, "speaker": "Speaker_01"},
            ],
        )
        diarizer.diarize.assert_called_once_with("unused.wav", 16000, 2)


if __name__ == "__main__":
    unittest.main()
