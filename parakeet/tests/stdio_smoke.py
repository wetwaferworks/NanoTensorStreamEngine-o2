import subprocess, time, sys, threading, queue
import os
exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "parakeet-c.exe")
p = subprocess.Popen([exe, "--stdio"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0, creationflags=0x08000000)
q = queue.Queue()
threading.Thread(target=lambda: [q.put(l.decode().rstrip()) for l in iter(p.stdout.readline, b"")], daemon=True).start()
t0 = time.time()
def wait(prefix, tmo=30):
    end = time.time() + tmo
    while time.time() < end:
        try: l = q.get(timeout=0.2)
        except queue.Empty: continue
        if not l.startswith("LEVEL"): print(f"{time.time()-t0:6.2f}s <- {l}")
        if l.startswith(prefix): return l
    print("TIMEOUT", prefix); return None
def send(c): print(f"{time.time()-t0:6.2f}s -> {c}"); p.stdin.write((c+"\n").encode()); p.stdin.flush()
wait("READY"); send("PING"); wait("PONG")
send("FILE " + os.path.join(os.environ.get("PARAKEET_MODEL", "models"), "samples", "espeak-test.wav")); wait("STAT")
send("START"); wait("REC"); time.sleep(2.6); send("STOP"); wait("DONE")
send("QUIT"); p.wait(5); print("exit", p.returncode)
