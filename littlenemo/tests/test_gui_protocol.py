#!/usr/bin/env python3
"""test_gui_protocol.py - end-to-end test of the v3 protocol paths
between the GUI client (NemoClient) and the fake engine fixture:
streamed thinking, image and video attachments (MEDIA / FRAME with
MEDIA_OK), engine-side refusals (MEDIA_ERR), every v3 telemetry line,
extended STATS, STOP mid-generation, a text-only Little Nemo, a
v2-only engine (backward compatibility), and the TCP transport.
Events are collected through a queue, the way the GUI does."""
import os
import queue
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gui"))
import nemo_chat  # noqa: E402
import nemo_media  # noqa: E402
import nemo_stats  # noqa: E402
from PIL import Image  # noqa: E402

FAKE = os.path.join(HERE, "fake_engine.py")
fails = []


def expect(cond, name):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


class Conn:
    def __init__(self, args, tcp_port=None):
        self.q = queue.Queue()
        self.client = nemo_chat.NemoClient(lambda k, p: self.q.put((k, p)))
        self.proc = None
        if tcp_port:
            self.proc = subprocess.Popen([sys.executable, FAKE, "--port",
                                          str(tcp_port)] + args)
            deadline = time.time() + 10
            while True:
                try:
                    self.client.connect_tcp("127.0.0.1", tcp_port)
                    break
                except OSError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.1)
        else:
            self.client.connect_process([sys.executable, FAKE, "--stdio"] + args)
        self.ready = self.until("ready")[-1][1]

    def until(self, kind, timeout=20):
        out = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                k, p = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            out.append((k, p))
            if k == kind:
                return out
        raise TimeoutError(f"no {kind} within {timeout}s; got "
                           f"{[k for k, _ in out][-5:]}")

    def close(self):
        self.client.close()
        if self.proc:
            self.proc.terminate()


def main():
    tmp = tempfile.mkdtemp(prefix="nemo-proto-")
    png = os.path.join(tmp, "photo one.png")
    Image.new("RGB", (40, 30), (0, 90, 200)).save(png)
    gif = os.path.join(tmp, "clip.gif")
    fr = [Image.new("RGB", (16, 16), (i * 50, 0, 0)) for i in range(4)]
    fr[0].save(gif, save_all=True, append_images=fr[1:], duration=40)
    img = nemo_media.inspect_attachment(png, "image")
    vid = nemo_media.inspect_attachment(gif, "video")
    frames, _ = nemo_media.extract_frames(vid, max_frames=4)
    vid["frames_list"] = frames

    # 1. SuperNemo fixture: v3, media, full telemetry
    c = Conn(["--model", "supernemo", "--delay", "0"])
    expect(nemo_chat.ready_media(c.ready) == {"image", "video"},
           "READY advertises image and video")
    expect(c.ready.get("fixture") == "1", "fixture labels itself in READY")
    ok, _ = nemo_chat.check_send("supernemo", True, True, c.ready, "look",
                                 [img, vid])
    expect(ok, "check_send allows media turn")
    ids = c.client.send_turn("what do you see?", [img, vid])
    ev = c.until("done")
    kinds = [k for k, _ in ev]
    expect(ids == [1, 2], "media ids assigned in order")
    oks = [p for k, p in ev if k == "media_ok"]
    expect([p["id"] for p in oks] == [1, 2], "MEDIA_OK for both attachments")
    expect(oks[1]["tokens"] == 64 * 4, "video accepted after all its FRAMEs")
    for k in ("prefill", "think", "tok", "layer", "tick", "expand", "dtc",
              "hs", "mem", "moe", "stat", "stats"):
        expect(k in kinds, f"{k} events flow")
    expect(kinds.index("think") < kinds.index("tok"), "THINK before TOK")
    expect(kinds[-2:] == ["stats", "done"], "STATS then DONE last")
    answer = "".join(p for k, p in ev if k == "tok")
    expect("test fixture" in answer and "2 attachment(s)" in answer,
           "answer mentions the fixture and both attachments")
    think = "".join(p for k, p in ev if k == "think")
    expect(nemo_stats.thinking_headline(think) != "", "thinking has a headline")
    st = [p for k, p in ev if k == "stats"][0]
    expect(isinstance(st["ttft_ms"], float) and st["prompt_tokens"] > 256,
           "extended STATS typed, prompt tokens include media tokens")
    m = nemo_stats.StatsModel()
    m.turn_begin()
    for k, p in ev:
        m.on_event(k, p)
    rows = {s: dict(i) for s, i in m.rows()}
    expect(rows["Now"]["In DTC"] != nemo_stats.DASH
           and rows["Tensor Handshake"]["Pass / ordinal"] != nemo_stats.DASH,
           "stats model fills DTC and handshake rows from the stream")

    # 2. engine-side refusal: a file that changed size / vanished
    bad = dict(img, bytes=img["bytes"] + 1)
    c.client.send_turn("again", [bad])
    ev = c.until("done")
    errs = [p for k, p in ev if k == "media_err"]
    expect(len(errs) == 1 and "size" in errs[0]["reason"],
           "MEDIA_ERR when the size does not match")
    gone = dict(img, path=os.path.join(tmp, "nope.png"))
    c.client.send_turn("again", [gone])
    ev = c.until("done")
    expect(any(k == "media_err" and "not found" in p["reason"]
               for k, p in ev), "MEDIA_ERR when the file is missing")
    c.client._send("MEDIA id=x type=image bytes=1 path=a")
    expect(c.until("media_err")[-1][1]["reason"].startswith("id and bytes"),
           "MEDIA_ERR for malformed MEDIA")

    # 3. STOP mid-generation
    c2 = Conn(["--model", "supernemo", "--delay", "0.05"])
    c2.client.send_turn("long one", [])
    c2.until("think")
    t0 = time.time()
    c2.client.stop()
    ev = c2.until("done", timeout=5)
    st = [p for k, p in ev if k == "stats"]
    expect(st and st[0]["stopped"] == 1 and time.time() - t0 < 2.0,
           "STOP ends the turn promptly with stopped=1")
    c2.client.reset()
    c2.client.send_turn("after reset", [])
    expect("done" in [k for k, _ in c2.until("done")], "turn after RESET")
    c2.close()

    # 4. Little Nemo fixture: text-only like the real engine
    c3 = Conn(["--model", "littlenemo", "--delay", "0"])
    expect(nemo_chat.ready_media(c3.ready) == set(),
           "Little Nemo READY has no media")
    ok, msg = nemo_chat.check_send("littlenemo", True, True, c3.ready, "hi",
                                   [img])
    expect(not ok and "text-only" in msg, "GUI refuses Little Nemo attachments")
    c3.client._send(nemo_chat.format_media_lines(9, img)[0])
    expect("text-only" in c3.until("media_err")[-1][1]["reason"],
           "a text-only engine refuses MEDIA on its side too")
    c3.close()

    # 5. v2-only engine: no v3 lines, GUI paths still work
    c4 = Conn(["--model", "littlenemo", "--v2", "--delay", "0"])
    c4.client.send_prompt("plain v2")
    ev = c4.until("done")
    kinds = {k for k, _ in ev}
    expect(not kinds & {"hw", "prefill", "expand", "dtc", "hs", "mem", "moe",
                        "stat"}, "v2 engine sends no v3 lines")
    m = nemo_stats.StatsModel()
    m.turn_begin()
    for k, p in ev:
        m.on_event(k, p)
    rows = {s: dict(i) for s, i in m.rows()}
    expect(rows["Throughput"]["Time to first token"] != nemo_stats.DASH,
           "TTFT still shown for a v2 engine (GUI-measured)")
    c4.close()

    # 6. TCP transport
    c5 = Conn(["--model", "supernemo", "--delay", "0"], tcp_port=17941)
    c5.client.send_turn("over tcp", [img])
    ev = c5.until("done")
    expect(any(k == "media_ok" for k, _ in ev), "MEDIA over TCP")
    c5.close()
    c.close()

    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
