# nq2-gui.py ... NanoQuatTwo console: a graphical front end for
# the C tools. GUI only, by JD's ruling: Python is permitted for
# this front end because it does no model work itself; every
# action is carried out by the C programs (nq2-run, and later the
# C inference engine) through their command-line interfaces.
#
# Written by Elfie (E.L.F.I.E.). Direction by JD.
#
# What it does today: open a packed model (.nq2 + .nq2book),
# show its vital signs, stream-verify the full expansion with
# the C runner (sizes, peak resident memory, throughput, and the
# FNV-1a checksum of every reconstructed weight), and peek at
# any single block, expanded on demand by the runner.
#
# What it does not do yet: chat. Talking to the model needs the
# C inference path (tokenizer, embedding codec, diffusion
# sampler), which is on the NanoQuatTwo road map. When that
# engine lands, this window gains a conversation pane that calls
# it the same way this one calls the runner.
#
# Needs: Python 3 with Tkinter (standard on python.org builds),
# and nq2-run (or nq2-run.exe on Windows) in the same folder or
# on the PATH.

import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "nq2-run.exe" if os.name == "nt" else "nq2-run")
if not os.path.exists(RUNNER):
    RUNNER = "nq2-run.exe" if os.name == "nt" else "nq2-run"


class Console(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NanoQuatTwo Console")
        self.geometry("760x560")
        self.stream_path = tk.StringVar()
        self.book_path = tk.StringVar()
        self.peek_block = tk.StringVar(value="0")
        self._build()

    def _build(self):
        top = tk.Frame(self, padx=10, pady=8)
        top.pack(fill="x")
        tk.Label(top, text="Index stream (.nq2):").grid(row=0, column=0, sticky="e")
        tk.Entry(top, textvariable=self.stream_path, width=52).grid(row=0, column=1, padx=4)
        tk.Button(top, text="Browse...", command=self._pick_stream).grid(row=0, column=2)
        tk.Label(top, text="Memory file (.nq2book):").grid(row=1, column=0, sticky="e")
        tk.Entry(top, textvariable=self.book_path, width=52).grid(row=1, column=1, padx=4)
        tk.Button(top, text="Browse...", command=self._pick_book).grid(row=1, column=2)

        bar = tk.Frame(self, padx=10)
        bar.pack(fill="x")
        tk.Button(bar, text="Verify and Expand (C runner)",
                  command=self._verify).pack(side="left")
        tk.Label(bar, text="   Peek block:").pack(side="left")
        tk.Entry(bar, textvariable=self.peek_block, width=10).pack(side="left", padx=2)
        tk.Button(bar, text="Peek", command=self._peek).pack(side="left")

        self.out = scrolledtext.ScrolledText(self, padx=8, pady=8, wrap="word",
                                             font=("Consolas", 10))
        self.out.pack(fill="both", expand=True, padx=10, pady=8)
        self._say("NanoQuatTwo Console. Pick a packed model to begin.\n"
                  "Everything below the glass is done by the C runner;\n"
                  "this window only asks it questions.\n")

    def _say(self, text):
        self.out.insert("end", text)
        self.out.see("end")

    def _pick_stream(self):
        p = filedialog.askopenfilename(filetypes=[("NQ2 stream", "*.nq2"), ("All", "*.*")])
        if p:
            self.stream_path.set(p)
            guess = p + "book" if p.endswith(".nq2") else ""
            if guess and os.path.exists(guess) and not self.book_path.get():
                self.book_path.set(guess)

    def _pick_book(self):
        p = filedialog.askopenfilename(filetypes=[("NQ2 book", "*.nq2book"), ("All", "*.*")])
        if p:
            self.book_path.set(p)

    def _paths_ok(self):
        if not self.stream_path.get() or not self.book_path.get():
            messagebox.showwarning("NanoQuatTwo", "Pick both files first: the index "
                                   "stream (.nq2) and its memory file (.nq2book).")
            return False
        return True

    def _run_async(self, args, label):
        # Tkinter is not thread-safe: the worker thread must never
        # touch the widgets directly. emit() marshals every update
        # back onto the UI thread through after().
        def emit(text):
            self.after(0, self._say, text)

        def work():
            emit("\n--- %s ---\n" % label)
            try:
                proc = subprocess.run([RUNNER] + args, capture_output=True,
                                      text=True, timeout=3600)
                emit(proc.stdout or "")
                if proc.stderr:
                    emit(proc.stderr)
                if proc.returncode != 0:
                    emit("(runner exited with code %d)\n" % proc.returncode)
            except FileNotFoundError:
                emit("Could not find the C runner (%s). Build it with "
                     "build.bat or build.sh, and put it beside this script.\n" % RUNNER)
            except Exception as exc:  # keep the window honest and alive
                emit("Error: %s\n" % exc)
        threading.Thread(target=work, daemon=True).start()

    def _verify(self):
        if self._paths_ok():
            self._run_async([self.stream_path.get(), self.book_path.get()],
                            "stream verify: full expansion, checksum, throughput")

    def _peek(self):
        if not self._paths_ok():
            return
        try:
            n = int(self.peek_block.get())
        except ValueError:
            messagebox.showwarning("NanoQuatTwo", "Peek needs a block number.")
            return
        self._run_async([self.stream_path.get(), self.book_path.get(),
                         "peek", str(n)], "peek block %d" % n)


if __name__ == "__main__":
    Console().mainloop()
