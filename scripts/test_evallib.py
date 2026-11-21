# Copyright (c) 2025-2026 Sunet.
# Contributor: Kristofer Hallin
#
# This file is part of Sunet Scribe.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Checks for scripts/evallib.py: `uv run python scripts/test_evallib.py`.
"""

from evallib import align, align_ops, normalise, score, swedish_number_words


def sv(text):
    return normalise(text, "swedish")


def test_numbers():
    assert swedish_number_words(19) == ["nitton"]
    assert swedish_number_words(23) == ["tjugo", "tre"]
    assert swedish_number_words(100) == ["hundra"]
    assert swedish_number_words(223) == ["två", "hundra", "tjugo", "tre"]
    assert swedish_number_words(2019) == ["två", "tusen", "nitton"]
    assert swedish_number_words(1_000_000) is None

    assert sv("19") == sv("nitton")
    assert sv("2019") == sv("tvåtusen nitton") == sv("två tusen nitton")
    assert sv("223") == sv("tvåhundratjugotre")
    assert sv("100") == sv("etthundra") == sv("hundra")
    assert sv("007") == ["007"]


def test_ordinary_words_untouched():
    assert sv("Månens yta, utgörs av stenar!") == ["månens", "yta", "utgörs", "av", "stenar"]
    assert sv("en bil och ett hus") == ["en", "bil", "och", "ett", "hus"]
    assert normalise("19", None) == ["19"]


def test_align():
    assert align("a b c".split(), "a b c".split()) == (0, 0, 0)
    assert align("a b c".split(), "a x c".split()) == (1, 0, 0)
    assert align("a b c".split(), "a c".split()) == (0, 1, 0)
    assert align("a c".split(), "a b c".split()) == (0, 0, 1)
    assert [op for op, _, _ in align_ops("a b".split(), "a x".split())] == ["eq", "sub"]


def test_score():
    hyp = [
        {"t": "a", "s": 0.0, "e": 0.5, "c": 0.9},
        {"t": "x", "s": 5.0, "e": 5.5, "c": 0.2},   # in the gap
        {"t": "b", "s": 9.0, "e": 9.5, "c": 0.9},
    ]
    result = score(["a", "b"], hyp, [(2.0, 8.0)], margin=1.0)

    assert (result["sub"], result["del"], result["ins"]) == (0, 0, 1)
    assert result["gap_words"] == 1 and result["gap_text"] == "x"
    assert result["confusions"][("", "x")] == 1


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
