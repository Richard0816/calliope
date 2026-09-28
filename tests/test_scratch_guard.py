"""The batch scratch janitor must never delete folders it didn't create.

Regression: a scratch dir pointed at the Calliope checkout made the
orphan sweep rmtree ``src/``, ``docs/`` and ``tests/``.
"""

from pathlib import Path

from calliope.tabs.batch.logic import (
    SCRATCH_OWNER_MARKER, is_scratch_mirror_skippable, is_scratch_owned,
    mark_scratch_owned, scratch_root_problem,
)


def test_project_checkout_rejected(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    assert "project folder" in scratch_root_problem(tmp_path)


def test_git_checkout_rejected(tmp_path):
    (tmp_path / ".git").mkdir()
    assert scratch_root_problem(tmp_path) is not None


def test_filesystem_root_rejected():
    assert scratch_root_problem(Path(Path.cwd().anchor)) is not None


def test_home_rejected():
    assert scratch_root_problem(Path.home()) is not None


def test_ancestor_of_protected_rejected(tmp_path):
    pkg = tmp_path / "code" / "src" / "calliope"
    pkg.mkdir(parents=True)
    assert scratch_root_problem(tmp_path, protected=(pkg,)) is not None
    assert scratch_root_problem(pkg, protected=(pkg,)) is not None


def test_dedicated_scratch_ok(tmp_path):
    scratch = tmp_path / "scratch"
    other = tmp_path / "out"
    assert scratch_root_problem(scratch, protected=(other, "")) is None


def test_ownership_marker(tmp_path):
    rec = tmp_path / "rec1"
    rec.mkdir()
    assert not is_scratch_owned(rec)
    mark_scratch_owned(rec)
    assert is_scratch_owned(rec)
    assert not is_scratch_owned(tmp_path / "missing")


def test_marker_not_mirrored(tmp_path):
    assert is_scratch_mirror_skippable(Path(SCRATCH_OWNER_MARKER), tmp_path)
