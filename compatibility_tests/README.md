# Existing editor compatibility

These optional tests use the actual Scribe UI consumer classes with synthetic
worker JSON results: legacy output, the additive raw-interval output from #47,
and the word-aligned output from #48. They do not require a backend account,
recording, model download, or inference.

Install the UI checkout's locked dependencies, then run from this worker repo:

```sh
SCRIBE_UI_SOURCE=/path/to/scribe-ui \
  /path/to/scribe-ui/.venv/bin/python -m pytest compatibility_tests/test_scribe_ui.py -q
```

Validated with SUNET/scribe-ui commit
`af622f29c0a54c955606866b8dfbb64041f8dadb` and its locked dependencies.
All six consumer tests pass. The UI's existing `tests/test_srt.py` also passes
alongside these checks: 34 tests total.

The checks cover unchanged import/export behavior when extra fields are added,
speaker-control construction with Unknown labels, word-payload loading, text
editing, splitting, merging, JSON save/reload, and SRT/VTT/RTF exports. Only the
browser JavaScript transport is stubbed; parsing and editor operations run in
the real consumer classes. This is not a browser or authenticated end-to-end
test and does not establish recognition or diarization accuracy.

The current UI safely ignores raw intervals, including on save. It ignores
`active_speakers` too, and adjacent captions with the same primary speaker can
be merged on import. Tests explicitly record the first limitation; neither
extra-field preservation nor overlap display is introduced by these worker PRs.
