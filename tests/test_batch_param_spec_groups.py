"""Batch union PARAM_SPEC: numbered group labels in pipeline order, and
no Tab 4 slider-bound knobs (they have no effect in a batch run)."""

from __future__ import annotations

import re

from calliope.tabs.batch.logic import build_batch_param_spec


def test_every_group_has_a_stage_prefix_in_order():
    spec = build_batch_param_spec()
    stages = []
    for entry in spec:
        m = re.match(r"(\d+)\. ", entry["group"])
        assert m, f"unnumbered group {entry['group']!r} ({entry['name']})"
        stages.append(int(m.group(1)))
    assert stages == sorted(stages)


def test_cellpose_sam_group_is_numbered():
    groups = {e["name"]: e["group"] for e in build_batch_param_spec()}
    assert groups["enable_sam_vcorr_pass"] == "2. Detection - Cellpose-SAM"


def test_slider_bounds_excluded():
    names = {e["name"] for e in build_batch_param_spec()}
    assert "cutoff_min" not in names
    assert "cutoff_max" not in names
    assert "cutoff_hz" in names
    assert "filter_order" in names
