#!/usr/bin/env python3
"""mock_engine.py - TEST FIXTURE ONLY, not part of the product.
Speaks PROTOCOL.md v2 well enough to exercise the GUI client headlessly:
streams a fake reasoning trace, layer ticks, telemetry, then an answer.

    python3 mock_engine.py PORT
"""
import socket
import sys
import time

THINK_WORDS = "let me think about this step by step first".split()
ANSWER_WORDS = "the answer is forty two and here is why".split()
TYPES = ["mamba", "mlp", "attention"]


def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n")


def run(conn):
    f = conn.makefile("rw", encoding="utf-8", newline="\n")
    f.write("READY nemo-engine 2 model=mock-nemo weights=raw layers=56\n")
    f.flush()
    for line in f:
        line = line.strip()
        if line.startswith("PROMPT"):
            tok = 0
            for i, w in enumerate(THINK_WORDS):
                tok += 1
                layer = i % 56
                f.write(f"LAYER {layer} {TYPES[layer % 3]}\n")
                f.write(f"THINK {esc(w)} \n")
                f.write(f"TICK mode=thinking layer={layer} next={layer + 1} "
                        f"active=1 staged=2 tok={tok} ms_tok=210.0 gbs=84.8\n")
                f.flush()
                time.sleep(0.005)
            for i, w in enumerate(ANSWER_WORDS):
                tok += 1
                layer = (len(THINK_WORDS) + i) % 56
                f.write(f"LAYER {layer} {TYPES[layer % 3]}\n")
                f.write(f"TOK {esc(w)} \n")
                f.write(f"TICK mode=answering layer={layer} next={layer + 1} "
                        f"active=1 staged=2 tok={tok} ms_tok=190.0 gbs=93.7\n")
                f.flush()
                time.sleep(0.005)
            f.write(f"STATS tokens={tok} ms=3400.0 tps=5.0 ms_per_tok=200.0 "
                    f"gbs_avg=89.0 peak_mb=512.0\n")
            f.write("DONE\n")
            f.flush()
        elif line == "QUIT":
            return


def main():
    port = int(sys.argv[1])
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    with conn:
        run(conn)


if __name__ == "__main__":
    main()
