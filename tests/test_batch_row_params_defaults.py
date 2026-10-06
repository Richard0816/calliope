"""Blank / partial batch rows must fall back to the batch defaults, not
inherit the previous row's overrides.

``_apply_row_params_to_tabs`` used to return early on an empty
``row.params`` and otherwise push only the row's own keys, while every
tab's ``apply_batch_row`` leaves keys it is not given untouched. A CSV
row with a blank (or partial) ``params`` cell therefore ran with
whatever the previous row had set. The batch now pushes the full
effective dict (spec defaults <- default_params <- row.params).
"""

from __future__ import annotations

import types

from calliope.gui_common import spec_defaults
from calliope.tabs.batch.tab import BatchTab


class _RecordingTab:
    """Mimics a tab's apply_batch_row: copy only the keys it is given."""

    def __init__(self):
        self.state: dict = {}

    def apply_batch_row(self, params, log=None):
        for k, v in params.items():
            self.state[k] = v


def _stub(default_params=None):
    s = types.SimpleNamespace()
    s.PARAM_SPEC = BatchTab.PARAM_SPEC
    s.default_params = (dict(default_params) if default_params is not None
                        else spec_defaults(BatchTab.PARAM_SPEC))
    s._app = types.SimpleNamespace(lowpass_tab=_RecordingTab())
    s._append_log = lambda msg: None
    s._effective_row_params = (
        lambda row: BatchTab._effective_row_params(s, row))
    return s


def _row(params):
    return types.SimpleNamespace(params=params)


def test_blank_row_does_not_inherit_previous_override():
    s = _stub()
    tab = s._app.lowpass_tab
    BatchTab._apply_row_params_to_tabs(s, _row({"cutoff_hz": 3.5}))
    assert tab.state["cutoff_hz"] == 3.5

    BatchTab._apply_row_params_to_tabs(s, _row({}))
    assert tab.state["cutoff_hz"] == s.default_params["cutoff_hz"]


def test_partial_row_fills_missing_keys_from_defaults():
    s = _stub()
    tab = s._app.lowpass_tab
    BatchTab._apply_row_params_to_tabs(
        s, _row({"cutoff_hz": 3.5, "filter_order": 4}))
    BatchTab._apply_row_params_to_tabs(s, _row({"cutoff_hz": 2.0}))
    assert tab.state["cutoff_hz"] == 2.0
    assert tab.state["filter_order"] == s.default_params["filter_order"]


def test_default_params_override_spec_defaults_and_fill_gaps():
    # An old calliope_batch.json may carry default_params missing newer
    # keys; those fall back to the spec defaults.
    s = _stub(default_params={"cutoff_hz": 1.7})
    eff = BatchTab._effective_row_params(s, _row({}))
    assert eff["cutoff_hz"] == 1.7
    spec = spec_defaults(BatchTab.PARAM_SPEC)
    assert eff["filter_order"] == spec["filter_order"]
    assert set(spec) <= set(eff)
