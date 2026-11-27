"""nemo_proc.py - how the GUI starts child processes (no Tk).

A GUI that starts console programs normally makes Windows open a console
window for each one, which lands on top of the GUI and steals focus. By
default nothing here does: every child is started with CREATE_NO_WINDOW and
its output is read through pipes, so the GUI can show it in its own log. The
"Show console windows" setting (littlenemo.ini, [gui] show_console) turns the
windows back on for the engine, and then its error output goes to that console
instead of the GUI. Helper programs (ffmpeg, nvidia-smi, the dictation
runtime) never get a window.

On other systems there are no console windows and these helpers do nothing.
"""
import os

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010

_show_console = False


def set_show_console(on):
    """Choose whether the engine gets a visible console window."""
    global _show_console
    _show_console = bool(on)


def console_hidden():
    """True when engine output is read through pipes into the GUI."""
    return not _show_console


def hidden_kwargs():
    """subprocess keyword arguments for a helper that must never show a
    window, whatever the setting."""
    return {"creationflags": CREATE_NO_WINDOW} if os.name == "nt" else {}


def engine_kwargs():
    """subprocess keyword arguments for the engine or converter: hidden by
    default, a new visible console when 'Show console windows' is on."""
    if os.name != "nt":
        return {}
    return {"creationflags":
            CREATE_NEW_CONSOLE if _show_console else CREATE_NO_WINDOW}


def parse_bool(text, default=False):
    """A settings-file switch: true/false, on/off, yes/no, 1/0."""
    t = str(text or "").strip().lower()
    if t in ("1", "true", "on", "yes"):
        return True
    if t in ("0", "false", "off", "no"):
        return False
    return default
