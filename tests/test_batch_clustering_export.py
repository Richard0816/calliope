"""The batch clustering stage must fail the stage when the cluster
export raises.

Tab 6's Export button handler (``_on_export``) swallows write errors
behind a modal dialog, so the batch now calls the raising
``_export_clusters`` directly. The deferred ``_fail_stage`` callback
also has to bind the exception at definition time: Python clears the
``except ... as e`` name when the block exits, so a plain
``lambda: ...(e)`` raised NameError when Tk later ran it.

These tests drive ``_stage_clustering`` unbound against stubs (no Tk).
"""

from __future__ import annotations

import types

from calliope.tabs.batch.tab import BatchTab


class _ClusterStub:
    def __init__(self, fail: bool):
        self.fail = fail
        self.summary_written = False
        self.status = None
        self.status_var = types.SimpleNamespace(
            set=lambda v: setattr(self, "status", v))

    def _export_clusters(self):
        if self.fail:
            raise OSError("disk full")
        return "OUT_DIR"

    def _on_export(self):  # must not be used by the batch
        raise AssertionError("batch called the dialog-swallowing handler")

    def _write_summary(self):
        self.summary_written = True

    def _on_run(self):
        pass


class _StageStub:
    def __init__(self, cl):
        self._app = types.SimpleNamespace(clustering_tab=cl)
        self._current_stage = "clustering"
        self.logs: list[str] = []
        self.deferred: list = []
        self.failed: list[tuple] = []
        self.done: list[tuple] = []
        self._subscribed = None
        self.state = types.SimpleNamespace(
            subscribe_clusters_ready=self._subscribe,
            subscribe_stage_error=lambda cb: (lambda: None))

    def _subscribe(self, cb):
        self._subscribed = cb
        return lambda: None

    def after(self, delay, fn):
        # Defer every callback, like Tk does: by the time it runs, the
        # ``except`` block that scheduled it has exited.
        self.deferred.append(fn)

    def _append_log(self, msg):
        self.logs.append(msg)

    def _fail_stage(self, stage, exc):
        self.failed.append((stage, exc))

    def _on_stage_done(self, stage, hint):
        self.done.append((stage, hint))


def _run(stub):
    BatchTab._stage_clustering(stub)
    stub._subscribed("PLANE0")
    for fn in list(stub.deferred):
        fn()


def test_export_failure_fails_stage():
    cl = _ClusterStub(fail=True)
    stub = _StageStub(cl)
    _run(stub)
    assert stub.done == []
    assert len(stub.failed) == 1
    stage, exc = stub.failed[0]
    assert stage == "clustering"
    assert isinstance(exc, OSError) and "disk full" in str(exc)


def test_export_success_advances_and_writes_summary():
    cl = _ClusterStub(fail=False)
    stub = _StageStub(cl)
    _run(stub)
    assert stub.failed == []
    assert stub.done == [("clustering", "PLANE0")]
    assert cl.summary_written
    assert cl.status == "Exported clusters to OUT_DIR"
