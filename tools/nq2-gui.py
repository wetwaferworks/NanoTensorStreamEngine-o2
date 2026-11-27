# nq2-gui.py ... NanoQuatTwo console: a graphical front end for
# the C tools. GUI only, by JD's ruling: Python is permitted for
# this front end because it does no model work itself; every
# action is carried out by the C programs (nq2-run, and later the
# C inference engine) through their command-line interfaces.
#
# Written by Elfie (E.L.F.I.E.). Direction by JD. The Quantize dialog is
# co-implemented by Cortana, built with Grok Bot by SpaceX.
#
# What it does today: open a packed model (.nq2 + .nq2book),
# show its vital signs, stream-verify the full expansion with
# the C runner (sizes, peak resident memory, throughput, and the
# FNV-1a checksum of every reconstructed weight), and peek at
# any single block, expanded on demand by the runner.
#
# Quantize... opens a dialog that runs the C converter (nq2-quantize) on a
# .safetensors file or a folder of them: pick the source and an output path,
# choose a preset or the converter options, and set the thread count (1 to
# 12, default all; the output is byte-identical at any count). Progress and
# the converter's output show in the dialog; Stop ends the run, and the
# converter publishes its pack only when it finishes. Nothing is downloaded,
# and the logic lives in nq2_quant.py.
#
# Console windows stay hidden (the runner and the converter are started
# without one) unless "Show console windows" is ticked.
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
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nq2_quant as nq  # noqa: E402

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
        self.settings = nq.load_settings()
        self.show_console = tk.BooleanVar(
            value=bool(self.settings.get("show_console", False)))
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._close)

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
        tk.Button(bar, text="Quantize...",
                  command=self._quantize).pack(side="right")
        tk.Checkbutton(bar, text="Show console windows",
                       variable=self.show_console).pack(side="right", padx=8)

        self.out = scrolledtext.ScrolledText(self, padx=8, pady=8, wrap="word",
                                             font=("Consolas", 10))
        self.out.pack(fill="both", expand=True, padx=10, pady=8)
        self._say("NanoQuatTwo Console. Pick a packed model to begin.\n"
                  "Everything below the glass is done by the C runner;\n"
                  "this window only asks it questions.\n")

    def _quantize(self):
        QuantizeDialog(self)

    def _close(self):
        self.settings["show_console"] = bool(self.show_console.get())
        nq.save_settings(self.settings)
        self.destroy()

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
                kw = {}
                if os.name == "nt":
                    kw["creationflags"] = nq.process_flags(
                        self.show_console.get())
                proc = subprocess.run([RUNNER] + args, capture_output=True,
                                      text=True, timeout=3600, **kw)
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


class QuantizeDialog(tk.Toplevel):
    """Run nq2-quantize from a window. All work is done by the C converter;
    this class collects options, shows progress and reports errors."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Quantize with nq2-quantize")
        self.geometry("820x760")
        self.job = None
        self.q = queue.Queue()
        s = parent.settings
        o = nq.new_options()
        o.update({k: v for k, v in s.get("options", {}).items() if k in o})
        o["threads"] = nq.clean_threads(s.get("threads", o["threads"]))
        self.v = {}
        for key, val in o.items():
            if isinstance(val, bool):
                self.v[key] = tk.BooleanVar(value=val)
            else:
                self.v[key] = tk.StringVar(value=str(val))
        self.src = tk.StringVar(value=s.get("source", ""))
        self.out_dir = tk.StringVar(value=s.get("out_dir", ""))
        self.out_name = tk.StringVar(value=s.get("out_name", "model"))
        self.exe = tk.StringVar(value=s.get("converter", ""))
        self.whole = tk.BooleanVar(value=False)
        self.preset = tk.StringVar(value=list(nq.PRESETS)[0])
        self._build()
        self.after(100, self._drain)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._refresh_source()

    # ---------------------------------------------------------- layout
    def _build(self):
        top = tk.Frame(self, padx=10, pady=8)
        top.pack(fill="x")
        top.columnconfigure(1, weight=1)

        def row(r, label, var, *btns):
            tk.Label(top, text=label).grid(row=r, column=0, sticky="e")
            tk.Entry(top, textvariable=var).grid(row=r, column=1,
                                                 sticky="ew", padx=4, pady=2)
            for i, (text, cmd) in enumerate(btns):
                tk.Button(top, text=text, command=cmd).grid(row=r,
                                                            column=2 + i)

        row(0, "Source (.safetensors):", self.src,
            ("File...", self._pick_file), ("Folder...", self._pick_folder))
        self.src_note = tk.Label(top, anchor="w", fg="#555")
        self.src_note.grid(row=1, column=1, sticky="w", padx=4)
        row(2, "Output folder:", self.out_dir, ("Browse...", self._pick_out))
        row(3, "Output name:", self.out_name)
        row(4, "Converter:", self.exe, ("Browse...", self._pick_exe))
        self.src.trace_add("write", lambda *_: self._refresh_source())

        opt = tk.LabelFrame(self, text="Options", padx=8, pady=6)
        opt.pack(fill="x", padx=10)
        r = 0
        tk.Label(opt, text="Preset:").grid(row=r, column=0, sticky="e")
        ttk.Combobox(opt, textvariable=self.preset, values=list(nq.PRESETS),
                     state="readonly", width=48).grid(row=r, column=1,
                                                      columnspan=3,
                                                      sticky="w", padx=4)
        tk.Button(opt, text="Apply preset",
                  command=self._apply_preset).grid(row=r, column=4)
        r += 1
        tk.Label(opt, text="Threads (1 to %d):" % nq.MAX_THREADS).grid(
            row=r, column=0, sticky="e")
        tk.Spinbox(opt, from_=1, to=nq.MAX_THREADS, width=5,
                   textvariable=self.v["threads"]).grid(row=r, column=1,
                                                        sticky="w", padx=4)
        tk.Label(opt, text="The output is byte-identical at any thread "
                 "count.", fg="#555").grid(row=r, column=2, columnspan=3,
                                           sticky="w")
        r += 1

        def field(label, key, col, width=10):
            nonlocal r
            tk.Label(opt, text=label).grid(row=r, column=col, sticky="e")
            e = tk.Entry(opt, textvariable=self.v[key], width=width)
            e.grid(row=r, column=col + 1, sticky="w", padx=4, pady=1)
            return e

        field("Block size:", "block", 0)
        field("Dictionary entries:", "entries", 2)
        r += 1
        self.budget_entry = field("Weight budget:", "budget", 0, 16)
        tk.Checkbutton(opt, text="Whole model (no cap)", variable=self.whole,
                       command=self._whole).grid(row=r, column=2,
                                                 columnspan=3, sticky="w")
        r += 1
        field("Sample blocks:", "sample_blocks", 0)
        field("Tensor sample blocks:", "tensor_sample_blocks", 2)
        r += 1
        field("k-means iterations:", "iters", 0)
        field("Tensor filter:", "filter", 2, 28)
        r += 1
        for label, key, vals, col in (
                ("Books:", "books", ("global", "per-tensor"), 0),
                ("Normalization:", "norm", ("minmax", "rms"), 2)):
            tk.Label(opt, text=label).grid(row=r, column=col, sticky="e")
            ttk.Combobox(opt, textvariable=self.v[key], values=vals,
                         state="readonly", width=12).grid(
                             row=r, column=col + 1, sticky="w", padx=4)
        r += 1
        tk.Label(opt, text="Format:").grid(row=r, column=0, sticky="e")
        ttk.Combobox(opt, textvariable=self.v["format"], values=("1", "2",
                     "3"), state="readonly", width=12).grid(
                         row=r, column=1, sticky="w", padx=4)
        tk.Label(opt, text="Product quantization:").grid(row=r, column=2,
                                                         sticky="e")
        ttk.Combobox(opt, textvariable=self.v["pq"], values=("0", "4", "8"),
                     state="readonly", width=12).grid(row=r, column=3,
                                                      sticky="w", padx=4)
        tk.Label(opt, text="(0 = off)", fg="#555").grid(row=r, column=4,
                                                        sticky="w")
        r += 1
        tk.Checkbutton(opt, text="Per-tensor budgets",
                       variable=self.v["budgets"]).grid(row=r, column=0,
                                                        columnspan=2,
                                                        sticky="w")
        tk.Checkbutton(opt, text="fp16 embeddings",
                       variable=self.v["embed_fp16"]).grid(row=r, column=2,
                                                           sticky="w")
        tk.Checkbutton(opt, text="Residual",
                       variable=self.v["residual"]).grid(row=r, column=3,
                                                         sticky="w")

        bar = tk.Frame(self, padx=10, pady=6)
        bar.pack(fill="x")
        self.btn_go = tk.Button(bar, text="Start", width=10,
                                command=self._start)
        self.btn_go.pack(side="left")
        self.btn_stop = tk.Button(bar, text="Stop", width=10, state="disabled",
                                  command=self._stop)
        self.btn_stop.pack(side="left", padx=4)
        tk.Button(bar, text="Show command",
                  command=self._show_command).pack(side="left", padx=4)
        self.btn_sum = tk.Button(bar, text="Checksums", state="disabled",
                                 command=self._checksums)
        self.btn_sum.pack(side="left", padx=4)
        tk.Button(bar, text="Close", command=self._close).pack(side="right")

        self.bar = ttk.Progressbar(self, maximum=100, mode="determinate")
        self.bar.pack(fill="x", padx=10)
        self.status = tk.Label(self, anchor="w", padx=10,
                               text="Ready. Pick a source and an output path.")
        self.status.pack(fill="x")
        self.log = scrolledtext.ScrolledText(self, wrap="word", height=12,
                                             font=("Consolas", 10))
        self.log.pack(fill="both", expand=True, padx=10, pady=8)

    # --------------------------------------------------------- helpers
    def _say(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def _set_status(self, text):
        self.status.configure(text=text)

    def _pick_file(self):
        p = filedialog.askopenfilename(
            filetypes=[("safetensors", "*.safetensors"), ("All", "*.*")])
        if p:
            self.src.set(p)

    def _pick_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.src.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory()
        if p:
            self.out_dir.set(p)

    def _pick_exe(self):
        p = filedialog.askopenfilename(
            filetypes=[("Program", "*.exe" if os.name == "nt" else "*"),
                       ("All", "*.*")])
        if p:
            self.exe.set(p)

    def _refresh_source(self):
        files = nq.collect_inputs(self.src.get()) if self.src.get() else []
        if not self.src.get():
            self.src_note.configure(text="")
        elif not files:
            self.src_note.configure(text="No .safetensors found here.")
        else:
            total = sum(os.path.getsize(f) for f in files)
            self.src_note.configure(
                text="%d shard(s), %.2f GB" % (len(files), total / 1e9))

    def _apply_preset(self):
        o = nq.preset_options(self.preset.get(),
                              nq.clean_threads(self.v["threads"].get()))
        for k, val in o.items():
            if k == "threads":
                continue
            var = self.v[k]
            var.set(val if isinstance(val, bool) else str(val))
        self._say("Preset applied: %s\n" % self.preset.get())

    def _whole(self):
        if self.whole.get():
            self.v["budget"].set("1000000000000")
            self.budget_entry.configure(state="disabled")
        else:
            self.v["budget"].set(str(nq.DEFAULTS["budget"]))
            self.budget_entry.configure(state="normal")

    def _collect(self):
        o = {}
        for k, var in self.v.items():
            val = var.get()
            if isinstance(val, bool):
                o[k] = val
            elif k in ("books", "norm", "filter"):
                o[k] = val
            else:
                try:
                    o[k] = int(str(val).strip())
                except ValueError:
                    o[k] = val   # validate() reports it by name
        return o

    def _prefix(self):
        d, n = self.out_dir.get().strip(), self.out_name.get().strip()
        return os.path.join(d, n) if d and n else ""

    def _plan(self):
        """(cmd, error text) after validating the whole form."""
        exe = nq.find_converter(self.exe.get().strip())
        inputs = nq.collect_inputs(self.src.get().strip())
        opts = self._collect()
        errs, warns = nq.validate(inputs, self._prefix(), opts)
        if not self.out_name.get().strip():
            errs.append("Give the output a name.")
        if exe is None:
            errs.append("Could not find nq2-quantize. Build it with build.bat "
                        "or build.sh and put it beside this script, or set "
                        "the Converter path.")
        if errs:
            return None, "\n".join(errs), warns
        return nq.build_command(exe, inputs, self._prefix(), opts), "", warns

    def _show_command(self):
        cmd, err, warns = self._plan()
        if cmd is None:
            self._say("Not ready:\n%s\n" % err)
            return
        self._say("Command:\n%s\n" % nq.quote_command(cmd))
        for w in warns:
            self._say("Warning: %s\n" % w)

    # ------------------------------------------------------------- run
    def _start(self):
        if self.job is not None:
            return
        cmd, err, warns = self._plan()
        if cmd is None:
            messagebox.showwarning("Quantize", err, parent=self)
            return
        if warns and not messagebox.askyesno(
                "Quantize", "\n\n".join(warns) + "\n\nContinue anyway?",
                parent=self):
            return
        pre = self._prefix()
        if any(os.path.exists(pre + e) for e in (".nq2", ".nq2book")):
            if not messagebox.askyesno(
                    "Quantize", "%s.nq2 or .nq2book already exists. The "
                    "converter will replace the pair when it finishes. "
                    "Continue?" % pre, parent=self):
                return
        self._save()
        self.log.delete("1.0", "end")
        self._say("Command:\n%s\n\n" % nq.quote_command(cmd))
        self.bar.configure(mode="indeterminate")
        self.bar.start(60)
        self.btn_go.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_sum.configure(state="disabled")
        self._set_status("Running (%s threads)..." % self.v["threads"].get())
        show = bool(self.parent.show_console.get())
        self.job = nq.Job(cmd, show_console=show)
        self.job.start(lambda t: self.q.put(("line", t)),
                       lambda p, ph: self.q.put(("progress", (p, ph))),
                       lambda rc, msg: self.q.put(("done", (rc, msg))))
        if show:
            self._say("Console window shown: the converter's output is "
                      "there, not here.\n")

    def _stop(self):
        if self.job is not None:
            self._set_status("Stopping...")
            self.job.stop()

    def _drain(self):
        try:
            while True:
                kind, data = self.q.get_nowait()
                if kind == "line":
                    self._say(data + "\n")
                elif kind == "progress":
                    pct, phase = data
                    if pct is not None:
                        self.bar.stop()
                        self.bar.configure(mode="determinate", value=pct)
                        self._set_status("%.1f%%" % pct)
                    elif phase:
                        self._set_status(phase)
                else:
                    self._finished(*data)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self._drain)

    def _finished(self, rc, msg):
        self.job = None
        self.bar.stop()
        self.bar.configure(mode="determinate", value=100 if rc == 0 else 0)
        self.btn_go.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self._set_status(msg)
        self._say("\n%s\n" % msg)
        if rc == 0:
            files = nq.output_files(self._prefix())
            for f in files:
                self._say("  %s  (%.1f MB)\n" % (f, os.path.getsize(f) / 1e6))
            self.btn_sum.configure(state="normal")
        elif "Could not start" in msg:
            messagebox.showerror("Quantize", msg, parent=self)

    def _checksums(self):
        pre = self._prefix()
        self.btn_sum.configure(state="disabled")
        self._say("Hashing %s ...\n" % pre)

        def work():
            for ext, h in nq.hash_outputs(pre).items():
                self.q.put(("line", "  sha256 %s  %s%s"
                            % (h, os.path.basename(pre), ext)))
        threading.Thread(target=work, daemon=True).start()

    def _save(self):
        s = self.parent.settings
        s["source"] = self.src.get()
        s["out_dir"] = self.out_dir.get()
        s["out_name"] = self.out_name.get()
        s["converter"] = self.exe.get()
        s["threads"] = nq.clean_threads(self.v["threads"].get())
        s["options"] = {k: v for k, v in self._collect().items()
                        if k != "threads"}
        s["show_console"] = bool(self.parent.show_console.get())
        nq.save_settings(s)

    def _close(self):
        if self.job is not None:
            if not messagebox.askyesno(
                    "Quantize", "A run is in progress. Stop it and close?",
                    parent=self):
                return
            self.job.stop()
        self._save()
        self.destroy()


if __name__ == "__main__":
    Console().mainloop()
