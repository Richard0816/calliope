"""The 15.08 Hz fallback frame rate is defined once (calliope_settings) and
re-exported as utils.DEFAULT_FPS; the suite2p ``fs`` default uses the same
value, so the three can't drift apart again."""
from calliope.core import calliope_settings, utils


def test_default_fps_single_source():
    assert utils.DEFAULT_FPS == calliope_settings.DEFAULT_FPS
    assert calliope_settings.CALLIOPE_BASE_SETTINGS["fs"] == utils.DEFAULT_FPS


def test_get_fps_falls_back_to_default(tmp_path):
    # No ops.npy and no notes_root -> the shared default.
    assert utils.get_fps_from_notes(str(tmp_path)) == utils.DEFAULT_FPS
