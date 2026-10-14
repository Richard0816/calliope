"""Drive the CalLIOPE GUI like a user, one command at a time.

usage: python gui_driver.py <workdir>   (Linux headless: xvfb-run -a python ...)
Then send commands with send.py.

Run under Xvfb. Every 300 ms the app checks CMD for a Python snippet,
executes it with helpers in scope, writes stdout/exception to OUT, then
deletes CMD. Message boxes are logged to POPUPS (and auto-answered);
file dialogs pop answers from ``NEXT_PATHS``.
"""
import contextlib
import io
import os
import sys
import time
import traceback
from pathlib import Path

if __name__ == "__main__":
    SP = Path(sys.argv[1])
    CMD, OUT, POPUPS = SP / "cmd.py", SP / "out.txt", SP / "popups.log"
    NEXT_PATHS: list = []
    T0 = time.time()


    def _log_popup(kind, title, msg):
        with open(POPUPS, "a") as f:
            f.write(f"[{time.time() - T0:7.1f}s] {kind}: {title!r}\n    "
                    + str(msg).replace("\n", "\n    ") + "\n")


    from tkinter import filedialog, messagebox  # noqa: E402

    for _k in ("showinfo", "showwarning", "showerror"):
        setattr(messagebox, _k, (lambda k: lambda title=None, message=None, **kw:
                                 _log_popup(k, title, message) or "ok")(_k))
    for _k in ("askyesno", "askokcancel", "askyesnocancel", "askquestion"):
        setattr(messagebox, _k, (lambda k: lambda title=None, message=None, **kw:
                                 _log_popup(k, title, message) or True)(_k))


    def _dialog(kind):
        def f(*a, **kw):
            ans = NEXT_PATHS.pop(0) if NEXT_PATHS else ""
            _log_popup(f"filedialog.{kind}", kw.get("title"), f"-> {ans}")
            return ans
        return f


    for _k in ("askdirectory", "askopenfilename", "askopenfilenames",
               "asksaveasfilename"):
        setattr(filedialog, _k, _dialog(_k))

    import customtkinter as ctk  # noqa: E402

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    from calliope.pipeline_gui import PipelineApp  # noqa: E402

    app = PipelineApp()
    app.geometry("1600x1000+0+0")
    TK_ERRORS = []


    def _hook(exc, val, tb):
        s = "".join(traceback.format_exception(exc, val, tb))
        TK_ERRORS.append(s)
        with open(POPUPS, "a") as f:
            f.write(f"[{time.time() - T0:7.1f}s] TK CALLBACK EXCEPTION\n{s}\n")


    app.report_callback_exception = _hook


    def shot(name):
        app.update()
        path = SP / "shots" / f"{name}.png"
        try:  # Windows / macOS / X11 with Pillow
            from PIL import ImageGrab
            x, y = app.winfo_rootx(), app.winfo_rooty()
            ImageGrab.grab(bbox=(x, y, x + app.winfo_width(),
                                 y + app.winfo_height())).save(path)
        except Exception:  # headless Xvfb without Pillow grab support
            os.system(f"import -window root {path}")
        return str(path)


    def tab(i):
        app._show_tab(i)
        app.update()


    def toplevels():
        return [w for w in app.winfo_children()
                if w.winfo_class() in ("Toplevel", "CTkToplevel")
                or type(w).__name__.endswith("Toplevel")]


    def widgets(root, cls=None, text=None):
        """Find descendant widgets by class name / button text."""
        out, stack = [], [root]
        while stack:
            w = stack.pop()
            stack.extend(w.winfo_children())
            if cls and type(w).__name__ != cls:
                continue
            if text is not None:
                try:
                    t = w.cget("text")
                except Exception:
                    continue
                if text not in str(t):
                    continue
            out.append(w)
        return out


    def click(root, text):
        btns = [b for b in widgets(root, text=text)
                if type(b).__name__ in ("CTkButton", "Button", "TButton",
                                        "CTkCheckBox", "CTkRadioButton")]
        if not btns:
            raise RuntimeError(f"no button {text!r}")
        b = btns[0]
        st = str(b.cget("state"))
        if st == "disabled":
            return f"button {text!r} is DISABLED"
        b.invoke()
        app.update()
        return f"clicked {text!r}"


    def all_texts(root):
        vals = []
        for w in widgets(root):
            try:
                t = w.cget("text")
                if t and type(w).__name__ in ("CTkLabel", "CTkButton"):
                    vals.append(f"{type(w).__name__}[{w.cget('state') if type(w).__name__=='CTkButton' else ''}]: {t}")
            except Exception:
                pass
        return vals


    def _poll():
        if CMD.exists():
            code = CMD.read_text()
            CMD.unlink()
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(code, globals())
            except Exception:
                buf.write("EXC:\n" + traceback.format_exc())
            OUT.write_text(buf.getvalue() + f"\n[t={time.time() - T0:.1f}s]\n")
        app.after(300, _poll)


    (SP / "shots").mkdir(exist_ok=True)
    app.after(500, _poll)
    app.mainloop()
