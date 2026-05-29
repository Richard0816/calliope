"""Smoke tests: every CalLIOPE submodule imports cleanly and the GUI
builds + cycles every tab without raising.

These tests are deliberately shallow -- they don't validate any
scientific output. Their job is to catch:

* a missing dependency in ``pyproject.toml`` (import would fail)
* a broken module after a refactor (import would fail)
* a tab that errors in ``__init__`` / ``_build_ui`` (headless GUI walk
  would raise via ``report_callback_exception``)

Run with: ``pytest tests/`` (requires ``pip install -e ".[dev]"``).
"""

from __future__ import annotations

import importlib
import pkgutil
import traceback
from pathlib import Path


def test_package_imports() -> None:
    """Walk the ``calliope`` package and import every submodule. Any
    missing dep or syntax error surfaces as a failed import."""
    import calliope
    pkg_path = Path(calliope.__file__).parent
    failures: list[str] = []
    for module in pkgutil.walk_packages([str(pkg_path)],
                                        prefix="calliope."):
        # Skip the ``scripts`` standalone CLI; it pulls in optional
        # cellfilter training deps that aren't part of the GUI surface.
        if module.name.startswith("calliope.scripts"):
            continue
        try:
            importlib.import_module(module.name)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{module.name}: {e!r}")
    assert not failures, "import failures:\n  " + "\n  ".join(failures)


def test_gui_builds_headless() -> None:
    """Build the GUI without showing the window, cycle every tab once,
    and tear it down. Catches any tab that regresses after a refactor.
    """
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    from calliope.pipeline_gui import PipelineApp

    app = PipelineApp()
    app.withdraw()  # don't pop a real window during the test

    errors: list[str] = []

    def _hook(exc, val, tb):
        errors.append("".join(traceback.format_exception(exc, val, tb)))

    app.report_callback_exception = _hook  # type: ignore[assignment]

    try:
        app.update_idletasks()
        for i in range(9):
            app._show_tab(i)
            app.update_idletasks()
            app.update()
    finally:
        try:
            app.destroy()
        except Exception:
            pass

    assert not errors, "runtime errors during tab walk:\n" + "\n---\n".join(
        errors)
