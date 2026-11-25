import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.settings import get_settings
from utils.subtitle_text import build_cues, merge_cues, wrap_caption
from utils.subtitle_timing import Cue, characters, retime, too_fast

settings = get_settings()
LINE = settings.SUBTITLE_LINE_LENGTH


def words(text, start=0.0, step=0.4, gaps=None):
    gaps = gaps or {}
    out, t = [], start

    for i, word in enumerate(text.split()):
        t += gaps.get(i, 0.0)
        out.append({"t": word, "s": t, "e": t + step - 0.05})
        t += step

    return out


def chunk(text, **kwargs):
    w = words(text, **kwargs)
    return {"start": w[0]["s"] - 0.5, "end": w[-1]["e"] + 0.5, "text": text, "words": w}


def test_short_caption_stays_on_one_line():
    assert wrap_caption("x" * LINE, LINE) == "x" * LINE


def test_both_lines_fit_and_bottom_is_not_shorter():
    text = "Jag tror att vi är mogna för det men vi måste prata först"
    top, bottom = wrap_caption(text, LINE).split("\n")
    assert len(top) <= LINE and len(bottom) <= LINE
    assert len(top) <= len(bottom) + 3


def test_break_after_punctuation_preferred():
    text = "Det var en gång en kung, och han hade tre söner som bodde"
    assert wrap_caption(text, LINE).split("\n")[0].endswith("kung,")


def test_no_break_after_article_or_preposition():
    top = wrap_caption("Vi gick hem efter att ha varit på stora marknaden i stan", LINE)
    assert top.split("\n")[0].split()[-1].lower() not in {"på", "i", "att", "en"}


def test_cue_fits_two_lines_and_max_duration():
    text = " ".join(["ord"] * 60)
    cues = build_cues([chunk(text, step=0.4)])
    assert len(cues) > 1

    for start, end, caption, _ in cues:
        lines = caption.split("\n")
        assert len(lines) <= 2 and all(len(l) <= LINE for l in lines)
        assert end - start <= settings.SUBTITLE_MAX_DURATION


def test_cue_without_words_is_still_cut():
    text = " ".join(["ord"] * 60)
    cues = build_cues([{"start": 0.0, "end": 30.0, "text": text, "words": []}])
    assert len(cues) > 1
    assert cues[0][0] == 0.0 and cues[-1][1] == 30.0


def test_cues_snap_to_words():
    [cue] = build_cues([chunk("Hej på dig")])
    assert abs(cue.start - 0.0) < 1e-9


def test_retime_no_overlap_and_limits():
    cues = [Cue(1.0, 1.2, "ok"), Cue(1.25, 2.0, "nästa"), Cue(5.0, 5.1, "x" * 80)]
    result = retime(cues)

    for a, b in zip(result, result[1:]):
        assert b.start - a.end >= settings.SUBTITLE_MIN_GAP - 1e-9

    for cue in result:
        assert 0 < cue.end - cue.start <= settings.SUBTITLE_MAX_DURATION


def test_different_speakers_get_dashes():
    cues = [Cue(0.0, 1.0, "Hur mår du?", "A"), Cue(1.3, 2.0, "Bra, tack.", "B")]
    [cue] = merge_cues(cues)
    assert cue.caption == "- Hur mår du?\n- Bra, tack."
    assert cue.speaker is None


def test_same_speaker_gets_no_dashes():
    cues = [Cue(0.0, 1.0, "Hur mår du?", "A"), Cue(1.3, 2.0, "Bra, tack.", "A")]
    [cue] = merge_cues(cues)
    assert cue.caption == "Hur mår du?\nBra, tack."


def test_unknown_speakers_get_no_dashes():
    cues = [Cue(0.0, 1.0, "Hur mår du?"), Cue(1.3, 2.0, "Bra, tack.")]
    assert merge_cues(cues)[0].caption == "Hur mår du?\nBra, tack."


def test_dashes_count_toward_line_length():
    long = "x" * (LINE - 1)
    cues = [Cue(0.0, 1.0, long, "A"), Cue(1.2, 2.0, "Ja.", "B")]
    assert len(merge_cues(cues)) == 2


def test_speaker_survives_build_cues():
    piece = dict(chunk("Hej på dig"), speaker="SPEAKER_1")
    assert build_cues([piece])[0].speaker == "SPEAKER_1"


def test_cue_is_held_long_enough_to_read():
    text = "x" * 30 + "\n" + "y" * 30
    [cue] = retime([Cue(1.0, 1.5, text)])
    assert characters(text) == 60
    assert (cue.end - cue.start) * settings.SUBTITLE_MAX_CPS >= 60
    assert not too_fast([cue])


def test_too_fast_when_next_cue_is_in_the_way():
    cues = [Cue(1.0, 1.5, "x" * 40), Cue(1.6, 3.0, "ok")]
    assert too_fast(retime(cues))


def test_start_earlier_when_next_cue_blocks_the_end():
    # 29 characters need 1.61 s at 18 characters per second; only 1.5 s is
    # there between the words and the next cue.
    cues = [Cue(10.0, 11.5, "x" * 29), Cue(11.6, 13.0, "ok")]
    [first, _] = retime(cues)
    assert not too_fast([first])
    assert first.start >= 10.0 - settings.SUBTITLE_MAX_EARLY - 1e-9


def test_borrowed_start_keeps_the_gap_to_the_cue_before():
    cues = [Cue(0.0, 1.0, "hej"), Cue(1.1, 2.6, "x" * 29)]
    first, second = retime(cues)
    assert second.start - first.end >= settings.SUBTITLE_MIN_GAP - 1e-9
    assert first.end > first.start


def test_long_words_never_overflow_a_line():
    text = " ".join(["Sammanfattningsvis"] * 4 + ["ja"] * 3)
    for _, _, caption, _ in build_cues([chunk(text)]):
        assert all(len(line) <= LINE for line in caption.split("\n"))
