"""Tests for the zero-event skip path in the GUI batch driver's
cross-correlation stage.

When event detection finds no population events, the xcorr tab's
``_on_run_per_event`` bails early (messagebox + return) WITHOUT publishing
``set_xcorr_ready``. The batch's ``_stage_crosscorrelation`` runs the
full-recording xcorr first, then per-event; its two-step completion
counter would never see the second publish, so the queue stalled forever
at the cross-correlation stage.

``_stage_crosscorrelation`` now checks ``xc._event_windows`` after the
full xcorr completes and, when empty, advances straight to stage-done
instead of kicking the per-event run. The full-recording xcorr (which
needs no events) still runs. These tests exercise the stage's completion
closure unbound against lightweight stubs -- no Tk required.
"""

from __future__ import annotations

import types

from calliope.tabs.batch.tab import BatchTab


class _XcorrStub:
    def __init__(self, event_windows):
        self._event_windows = event_windows
        self.full_called = False
        self.per_event_called = False

    def _on_run_full(self):
        self.full_called = True

    def _on_run_per_event(self):
        self.per_event_called = True


class _StageStub:
    """Minimal stand-in exposing only what _stage_crosscorrelation reads."""

    def __init__(self, xc):
        self._app = types.SimpleNamespace(xcorr_tab=xc)
        self._current_stage = "crosscorrelation"
        self._xcorr_step = None
        self.logs: list[str] = []
        self.stage_done: list[tuple] = []
        self.scheduled: list[tuple] = []   # (delay, fn) for nonzero delays
        self._subscribed = None
        self.state = types.SimpleNamespace(
            subscribe_xcorr_ready=self._subscribe)

    def _subscribe(self, cb):
        self._subscribed = cb

    def after(self, delay, fn):
        # Run zero-delay callbacks inline (that's the stage-done hop);
        # record deferred ones so the test can assert what was scheduled.
        if delay == 0:
            fn()
        else:
            self.scheduled.append((delay, fn))

    def _append_log(self, msg):
        self.logs.append(msg)

    def _on_stage_done(self, stage, hint):
        self.stage_done.append((stage, hint))


def test_skips_per_event_when_no_events():
    xc = _XcorrStub(event_windows=[])
    stub = _StageStub(xc)
    BatchTab._stage_crosscorrelation(stub)

    # The stage kicks the full-recording xcorr.
    assert any(d == 50 and fn == xc._on_run_full for d, fn in stub.scheduled)

    # Simulate the full-xcorr completion publish.
    stub._subscribed("FULL_DONE")

    # Per-event must NOT run, and the stage advances with the full payload.
    assert xc.per_event_called is False
    assert ("crosscorrelation", "FULL_DONE") in stub.stage_done


def test_runs_per_event_when_events_present():
    xc = _XcorrStub(event_windows=[(0.0, 1.0), (2.0, 3.0)])
    stub = _StageStub(xc)
    BatchTab._stage_crosscorrelation(stub)

    # First publish: full done -> per-event scheduled, stage not yet done.
    stub._subscribed("FULL_DONE")
    assert stub._xcorr_step == "per_event"
    assert any(d == 200 and fn == xc._on_run_per_event
               for d, fn in stub.scheduled)
    assert stub.stage_done == []

    # Second publish: per-event done -> stage advances.
    stub._subscribed("PER_EVENT_DONE")
    assert ("crosscorrelation", "PER_EVENT_DONE") in stub.stage_done
