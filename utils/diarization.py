"""Serialization of the original, overlap-preserving pyannote annotation."""


def normalize_speaker_name(speaker: str) -> str:
    """Use the same speaker identifiers as the transcript segments."""
    if speaker.startswith("SPEAKER_"):
        return "Speaker_" + speaker.removeprefix("SPEAKER_")
    return speaker


def serialize_diarization(annotation) -> list[dict]:
    """Keep every turn, including overlaps and turns without transcribed words.

    Times are absolute seconds. Do not round, merge, clip to transcript
    boundaries, or substitute the exclusive diarization annotation.
    """
    return [
        {
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": normalize_speaker_name(speaker),
        }
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
