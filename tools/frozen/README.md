# Frozen evidence

Files here are kept byte for byte as evidence. They are not compiled, not
a build base, and not to be edited or "fixed".

| File | Frozen at | SHA-256 (committed bytes) | What it is |
|---|---|---|---|
| `nq2-infer-triforce-dbg.04a4640.c.txt` | commit `04a4640` (2026-11-25) | `f09b813d32c201f609391f65de33e0547ef6a419357c6f73c0c6ef5d39f7c700` | the Triforce debug engine exactly as it stood when the anchor failure was bisected, both defects included; the hash `MANIFEST-SHA256.txt` records for `tools/nq2-infer-triforce-dbg.c` |

Check it with `git show HEAD:tools/frozen/nq2-infer-triforce-dbg.04a4640.c.txt | sha256sum`
(or `git show 04a4640:tools/nq2-infer-triforce-dbg.c | sha256sum`). A
Windows checkout with `core.autocrlf=true` writes CRLF to disk, so hash the
committed blob as shown, not the working file.

The `.c.txt` name keeps it out of every build and out of the zero-mutex
grep: the snapshot holds the pthread locks it had in 2026-11-24, which is
part of what it preserves.

The buildable debug tool is `tools/nq2-infer-triforce-dbg.c`. Commit
`ef39281` (2026-11-26, the zero-mutex Tensor Handshake conversion)
rewrote that file in place; it is what `build.sh` and `build.bat` compile
as `nq2-infer-triforce-dbg`. See the record note in
[PRECISION-GATE.md](../../docs/PRECISION-GATE.md#record-notes-2026-11-27).
