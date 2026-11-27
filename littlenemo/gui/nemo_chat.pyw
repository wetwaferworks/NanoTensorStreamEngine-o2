"""nemo_chat.pyw - start the Little Nemo chat window with no console.

Double-click this file, or run `pythonw gui\nemo_chat.pyw [options]`. It is the
same program as nemo_chat.py; the .pyw extension makes Windows use pythonw, so
no console window opens behind or in front of the chat window. The engine is
started without a window too (see nemo_proc.py). Anything the program would
have printed goes to gui.log in the NemoChat folder of your local app data
(or the temp folder), outside the repository.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if sys.stdout is None or sys.stderr is None:
    # pythonw has no console, so stdout and stderr start out as None.
    log_dir = os.path.join(os.environ.get("LOCALAPPDATA")
                           or os.environ.get("TEMP") or HERE, "NemoChat")
    try:
        os.makedirs(log_dir, exist_ok=True)
        log = open(os.path.join(log_dir, "gui.log"), "a", encoding="utf-8",
                   buffering=1)
    except OSError:
        log = open(os.devnull, "w")
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log

import nemo_chat  # noqa: E402

if __name__ == "__main__":
    sys.exit(nemo_chat.main(sys.argv[1:]))
