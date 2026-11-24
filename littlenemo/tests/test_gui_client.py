#!/usr/bin/env python3
"""test_gui_client.py - headless test of the GUI's protocol client
against the mock engine. Asserts the event stream's shape and ordering:
READY first, THINK before TOK, LAYER and TICK telemetry flowing, STATS
then DONE last, and TICK fields parsed to the right types.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gui"))
import nemo_chat  # noqa: E402

PORT = 17931
events = []
done = threading_event = None


def main():
    import threading

    done_ev = threading.Event()

    def on_event(kind, payload):
        events.append((kind, payload))
        if kind == "done":
            done_ev.set()

    mock = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "mock_engine.py"),
         str(PORT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        client = nemo_chat.NemoClient(on_event)
        deadline = time.time() + 10
        while True:
            try:
                client.connect_tcp("127.0.0.1", PORT)
                break
            except OSError:
                if time.time() > deadline:
                    raise
                time.sleep(0.1)
        client.send_prompt("what is the answer?")
        if not done_ev.wait(15):
            print("FAIL: no DONE within 15s")
            return 1
        client.quit()
    finally:
        mock.terminate()

    kinds = [k for k, _ in events]
    fails = []

    def expect(cond, name):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    expect(kinds[0] == "ready", "READY first")
    expect("think" in kinds and "tok" in kinds, "THINK and TOK present")
    expect(kinds.index("think") < kinds.index("tok"), "THINK before TOK")
    expect(kinds.count("layer") >= 10, f"LAYER ticks flow ({kinds.count('layer')})")
    expect(kinds.count("tick") >= 10, f"TICK telemetry flows ({kinds.count('tick')})")
    expect(kinds[-2] == "stats" and kinds[-1] == "done", "STATS then DONE last")
    ticks = [p for k, p in events if k == "tick"]
    expect(all(isinstance(t["tok"], int) and isinstance(t["gbs"], float)
               for t in ticks), "TICK fields typed (int tok, float gbs)")
    expect({t["mode"] for t in ticks} == {"thinking", "answering"},
           "both modes seen")
    think_text = "".join(p for k, p in events if k == "think")
    answer_text = "".join(p for k, p in events if k == "tok")
    expect("step by step" in think_text, "thinking text assembled")
    expect("forty two" in answer_text, "answer text assembled")
    stats = [p for k, p in events if k == "stats"][0]
    expect(stats["tokens"] == len(ticks), "STATS token count matches TICKs")
    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
