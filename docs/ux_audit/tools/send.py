"""Send one Python snippet to a running gui_driver.py and print its output.

usage: python send.py <workdir> "<python code>"
The code runs on the GUI thread with the driver's helpers in scope
(app, tab, click, shot, widgets, NEXT_PATHS, ...).
"""
import sys
import time
from pathlib import Path

wd, code = Path(sys.argv[1]), sys.argv[2]
out = wd / "out.txt"
out.unlink(missing_ok=True)
tmp = wd / "cmd.py.tmp"
tmp.write_text(code + "\n")
tmp.replace(wd / "cmd.py")
deadline = time.time() + 300
while not out.exists():
    if time.time() > deadline:
        sys.exit("TIMEOUT: app busy, frozen or dead")
    time.sleep(0.5)
print(out.read_text())
