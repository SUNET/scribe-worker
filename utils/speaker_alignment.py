"""Align transcript words to overlap-preserving diarization intervals.

Words remain indivisible. A word crossing a diarization boundary is assigned
by temporal overlap; this is not source separation or a guarantee about its
true speaker. Unresolvable words and chunks retain their text with Unknown.
"""

from bisect import bisect_right
from collections import Counter, defaultdict
from math import isclose, isfinite

from utils.words import average_confidence


def activity_spans(intervals):
    """Partition the timeline into spans with constant simultaneous speakers."""
    events = defaultdict(Counter)
    for interval in intervals:
        start, end = interval["start"], interval["end"]
        if not (isfinite(start) and isfinite(end) and end > start):
            continue
        speaker = interval["speaker"]
        events[start][speaker] += 1
        events[end][speaker] -= 1
    points = sorted(events)
    active = Counter()
    spans = []
    for index, point in enumerate(points[:-1]):
        active.update(events[point])
        speakers = tuple(sorted(s for s, count in active.items() if count > 0))
        spans.append((point, points[index + 1], speakers))
    return spans


def _activity(start, end, spans, ends):
    for index in range(bisect_right(ends, start), len(spans)):
        left, right, speakers = spans[index]
        if left >= end:
            break
        duration = min(right, end) - max(left, start)
        if duration > 0:
            yield duration, speakers


def _word_speakers(start, end, spans, ends):
    durations = defaultdict(float)
    covered = 0.0
    for duration, speakers in _activity(start, end, spans, ends):
        durations[speakers] += duration
        covered += duration
    # Silence outside the annotation also counts, rather than allowing a
    # tiny intersection with speech to label a word mostly in silence.
    durations[()] += max(0.0, end - start - covered)
    greatest = max(durations.values(), default=0.0)
    winners = [state for state, duration in durations.items()
               if isclose(duration, greatest, rel_tol=1e-9, abs_tol=1e-12)]
    return winners[0] if len(winners) == 1 else ()


def _text_boundaries(text, words):
    """Map every word to the original text without dropping punctuation.

    Require complete agreement ignoring whitespace. Otherwise splitting is
    unsafe: retained text must not be replaced with a partial word payload.
    """
    positions = [index for index, char in enumerate(text) if not char.isspace()]
    compact = "".join(text[index] for index in positions)
    tokens = ["".join(word["t"].split()) for word in words]
    if not tokens or any(not token for token in tokens) or "".join(tokens) != compact:
        return None
    boundaries = [0]
    consumed = 0
    for token in tokens[:-1]:
        consumed += len(token)
        boundaries.append(positions[consumed])
    boundaries.append(len(text))
    return boundaries


def _segment(start, end, text, speakers, confidence=None):
    result = {
        "start": float(start), "end": float(end), "text": text.strip(),
        "speaker": speakers[0] if len(speakers) == 1 else "Unknown",
        "active_speakers": list(speakers), "duration": float(end - start),
    }
    if confidence is not None:
        result["avg_score"] = confidence
    return result


def align_chunks(chunks, intervals):
    """Split chunks on word-level speaker-state changes, preserving all text.

    Multiple active speakers always refer to simultaneous activity in the
    annotation, never to the union of consecutive turns. The singular
    speaker is Unknown during overlap, ties, silence, or missing alignment.
    """
    spans = activity_spans(intervals)
    ends = [span[1] for span in spans]
    result = []
    for chunk in chunks:
        words = chunk.get("words", [])
        boundaries = _text_boundaries(chunk["text"], words)
        valid = boundaries is not None and all(
            isfinite(w["s"]) and isfinite(w["e"]) and w["e"] > w["s"]
            # The compact word payload is rounded to milliseconds, unlike
            # the original Whisper chunk boundaries.
            and chunk["start"] - 0.001 <= w["s"] <= w["e"] <= chunk["end"] + 0.001
            for w in words
        ) and all(a["s"] < b["s"] for a, b in zip(words, words[1:]))
        if not valid:
            # A single consistent speech state is usable without words.
            # Sequential speakers are not evidence of overlapping speech.
            states = {speakers for _, speakers in
                      _activity(chunk["start"], chunk["end"], spans, ends)
                      if speakers}
            speakers = next(iter(states)) if len(states) == 1 else ()
            result.append(_segment(chunk["start"], chunk["end"], chunk["text"],
                                   speakers, chunk.get("avg_score")))
            continue

        states = [_word_speakers(w["s"], w["e"], spans, ends) for w in words]
        groups = [0] + [i for i in range(1, len(words)) if states[i] != states[i - 1]]
        groups.append(len(words))
        cuts = [chunk["start"]]
        for index in groups[1:-1]:
            # Cut within a gap when possible, otherwise at the next word's
            # start. Never interpolate word timings from sentence length.
            cuts.append((min(words[index - 1]["e"], words[index]["s"])
                         + words[index]["s"]) / 2)
        cuts.append(chunk["end"])
        for group, (first, stop) in enumerate(zip(groups, groups[1:])):
            result.append(_segment(
                cuts[group], cuts[group + 1],
                chunk["text"][boundaries[first]:boundaries[stop]], states[first],
                average_confidence(words[first:stop]),
            ))
    return result
