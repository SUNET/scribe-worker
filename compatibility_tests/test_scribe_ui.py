"""Consumer checks for worker PRs #47/#48; synthetic worker output only."""
import copy
import json
import os
import sys
from pathlib import Path

import pytest
ui_source = os.environ.get("SCRIBE_UI_SOURCE")
if not ui_source:
    pytest.skip("Set SCRIBE_UI_SOURCE to a scribe-ui checkout", allow_module_level=True)
sys.path.insert(0, str(Path(ui_source).resolve()))

from nicegui import ui
from utils.srt import SRTEditor

FIXTURES = json.loads(Path(__file__).with_name("worker_results.json").read_text())


def editor_for(name):
    editor = SRTEditor('compatibility-test', 'txt', 'synthetic.wav')
    editor.parse_txt(json.dumps(FIXTURES[name]))
    return editor


def test_extra_fields_do_not_change_import_or_exports():
    before, after = editor_for('legacy'), editor_for('raw')
    assert before.export_json() == after.export_json()
    assert before.export_srt() == after.export_srt()
    assert before.export_vtt() == after.export_vtt()
    assert before.export_rtf(speakers=True, times=True, block_nr=True) == after.export_rtf(speakers=True, times=True, block_nr=True)


@pytest.mark.parametrize('name', ['legacy', 'raw', 'aligned'])
def test_editor_controls_editing_and_exports(name, monkeypatch):
    # Browser transport is outside this server-side consumer contract test.
    monkeypatch.setattr(ui, "run_javascript", lambda *args, **kwargs: None)
    editor = editor_for(name)
    editor.load_words(FIXTURES['words'])
    assert len(editor.words) == 4
    assert ' '.join(c.text for c in editor.captions) == FIXTURES[name]['full_transcription']
    with ui.column():
        for caption in editor.captions:
            # Exercise both display cards and editable speaker selectors.
            caption.is_selected = False
            editor.create_caption_card(caption)
            caption.is_selected = True
            editor.create_caption_card(caption)
    if name == 'aligned':
        assert [c.speaker for c in editor.captions] == [
            'Speaker_00', 'Speaker_01', 'Speaker_00', 'Unknown']
        assert 'Unknown' in editor.speakers
    editor.update_caption_text(editor.captions[0], 'Edited text here.')
    editor.split_caption(editor.captions[0], cursor_position=6)
    editor.merge_with_next(editor.captions[0])
    saved = editor.export_json()
    assert 'Edited' in saved['full_transcription']
    restored = SRTEditor('restored', 'txt', 'synthetic.wav')
    restored.parse_txt(json.dumps(saved))
    assert restored.captions
    assert editor.export_srt()
    assert editor.export_vtt().startswith('WEBVTT')
    assert editor.export_rtf(speakers=True, times=True, block_nr=True).startswith('{\\rtf')


def test_unknown_additional_fields_are_ignored_on_segments_too():
    payload = copy.deepcopy(FIXTURES['aligned'])
    for segment in payload['segments']:
        segment['future_metadata'] = {'ignored': True}
    editor = SRTEditor('extra', 'txt', 'synthetic.wav')
    editor.parse_txt(json.dumps(payload))
    assert editor.export_json() == editor_for('aligned').export_json()


def test_raw_intervals_are_ignored_not_preserved_on_save():
    # This is a documented limitation, not an import/export failure.
    assert 'diarization_segments' not in editor_for('raw').export_json()
