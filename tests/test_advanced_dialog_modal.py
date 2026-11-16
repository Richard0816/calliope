"""The Advanced dialog grabs input only once it is on screen (macOS left
typing with the main window when it grabbed during construction), and
status labels wrap to their width instead of running under buttons."""
import time

import pytest

ctk = pytest.importorskip("customtkinter")
from calliope import gui_common  # noqa: E402


def _pump(root, cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        root.update()
        time.sleep(0.02)
    return cond()


@pytest.fixture(scope="module")   # one Tk interpreter for the module
def root():
    try:
        r = ctk.CTk()
    except Exception as e:  # no display
        pytest.skip(f"no Tk display: {e}")
    yield r
    r.destroy()


def test_dialog_grabs_after_it_is_shown(root):
    specs = [{"name": "a", "type": "float", "default": 1.0},
             {"name": "b", "type": "bool", "default": True}]
    dlg = gui_common.AdvancedDialog(root, "t", specs, {})
    assert root.grab_current() is None          # not grabbed while building
    assert _pump(root, lambda: root.grab_current() is dlg)
    assert len(dlg._entries) == 1               # the text field gets focus
    dlg._on_cancel()


def test_fields_stay_visible_in_a_narrow_dialog(root):
    # Long labels and help text used to squeeze the entry column to zero
    # width once the dialog was narrower than label + help text.
    specs = [{"name": f"p{i}", "label": "a rather long parameter label " * 2,
              "type": "float", "default": 1.0, "group": "G",
              "help": "a long help sentence that would claim the width " * 4}
             for i in range(3)]
    dlg = gui_common.AdvancedDialog(root, "t", specs, {})
    dlg.geometry("480x400")
    _pump(root, lambda: False, timeout=1.0)
    right = dlg.winfo_rootx() + dlg.winfo_width()
    for e in dlg._entries:
        assert e.winfo_ismapped() and e.winfo_width() > 60
        assert e.winfo_rootx() + e.winfo_width() <= right
    dlg._on_cancel()


def test_label_wraps_to_its_width(root):
    lab = ctk.CTkLabel(root, text="x " * 400)
    lab.pack(fill="x")
    gui_common.wrap_to_width(lab)
    root.geometry("500x200")
    assert _pump(root, lambda: 0 < lab.cget("wraplength") <= 500)
