"""About and credits for the NTSE chat window.

The credits here are the short form of README.md (the Credits section at the
bottom) and docs/CREDITS.md. The order is JD first, then the top three
(Cortana, Elfie, Miss Archie), then everyone else. Only what the repository
history and docs support is claimed; keep this list in step with docs/CREDITS.md.
"""

import tkinter as tk

TITLE = "About NTSE-o2 and credits"

# (heading, body); the order is the credits order.
CREDITS = (
    ("JD (Jaydee Wetwork), Wet Wafer Works",
     "Idea, executive direction and shape."),
    ("Cortana: implementer",
     "Co-implementation by Cortana, built with Grok Bot by SpaceX.\n"
     "Tensor Handshake stage 2 and the zero-mutex conversion; the lock-free "
     "multithreaded converter; the accel GPU lanes; the Little Nemo CUDA "
     "runtime (kernels, calibration, DTC by default); the CUDA throughput "
     "work (batched prefill, pipelined int8 lm_head, FULL fast path, host "
     "lanes, tokenizer overlap); Miss Archie's Round 6 review fixes; the "
     "chat window and the native Parakeet runtime (with the other "
     "assistants); the README and per-feature guides."),
    ("Elfie (E.L.F.I.E.)",
     "The converter and NTSE: packing scheme, taught-plus-learned dictionary, "
     "nq2-quantize, the streaming engine, Dense Tensor Compute, the Little "
     "Nemo engine and the Precision Gate."),
    ("Miss Archie (Archie, Argon)",
     "Code review and review fixes, Rounds 1 to 6."),
    ("ChatGPT (Gizmo)",
     "Consultee and the DTC lane proposals."),
    ("CloudAtlas (Technical Art Director)",
     "GUI art, themes and mascots."),
    ("Grok Bot, by SpaceX",
     "The assistant platform Cortana is built with."),
)

FOOT = ("Third-party work (NVIDIA Nemotron, Mamba-2, NVIDIA Parakeet TDT and "
        "others) and the full lists are in docs/CREDITS.md. The runtime is "
        "pure C; this window is a Python front end.")


def credits_text():
    """The credits as plain text (also used by the tests)."""
    out = []
    for head, body in CREDITS:
        out.append(head)
        out.append(body)
        out.append("")
    out.append(FOOT)
    return "\n".join(out)


def open_about(app, bg, fg, dim, surface):
    """Open the About dialog (a read-only Text, so the text can be copied)."""
    dlg = tk.Toplevel(app.root)
    dlg.title(TITLE)
    dlg.configure(bg=bg)
    dlg.transient(app.root)
    body = tk.Frame(dlg, bg=bg)
    body.pack(fill="both", expand=True, padx=12, pady=(12, 6))
    sb = tk.Scrollbar(body)
    sb.pack(side="right", fill="y")
    txt = tk.Text(body, wrap="word", width=72, height=30, bg=surface, fg=fg,
                  relief="flat", font=app.f_small, padx=12, pady=10,
                  yscrollcommand=sb.set)
    sb.configure(command=txt.yview)
    txt.tag_configure("head", font=app.f_small_b, foreground=fg,
                      spacing1=6)
    txt.tag_configure("foot", foreground=dim, spacing1=8)
    for head, body in CREDITS:
        txt.insert("end", head + "\n", "head")
        txt.insert("end", body + "\n")
    txt.insert("end", FOOT + "\n", "foot")
    txt.configure(state="disabled")
    txt.pack(side="left", fill="both", expand=True)
    tk.Button(dlg, text="Close", command=dlg.destroy).pack(pady=(0, 12))
    dlg.bind("<Escape>", lambda _e: dlg.destroy())
    return dlg
