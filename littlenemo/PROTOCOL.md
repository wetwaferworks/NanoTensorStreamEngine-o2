# Little Nemo engine protocol v2 and v3

Protocol v2 is the base; v3 is a backward-compatible set of extensions
(2026-11-26), described in the second half of this file. The GUI-side view of
every message is in [docs/GUI.md](../docs/GUI.md#protocol-v3-message-reference).

## Message index

| Message | Direction | Since | Meaning |
|---|---|---|---|
| `PROMPT` | client to engine | v2 | one user turn |
| `STOP` | client to engine | v2 | end the current generation early |
| `RESET` | client to engine | v2 | clear the conversation (and pending attachments) |
| `QUIT` | client to engine | v2 | end the session |
| `MEDIA` | client to engine | v3 | announce one attachment for the next PROMPT |
| `FRAME` | client to engine | v3 | one extracted video frame for a MEDIA id |
| `READY` | engine to client | v2 (v3 adds `media`, `accel`, `moe`, `fixture`) | the engine is up |
| `THINK` | engine to client | v2 | a piece of the reasoning trace |
| `TOK` | engine to client | v2 | a piece of the visible answer |
| `LAYER` | engine to client | v2 | the forward pass entered a layer |
| `TICK` | engine to client | v2 | per-token telemetry |
| `STATS` | engine to client | v2 (v3 adds keys) | end-of-turn summary |
| `DONE` | engine to client | v2 | the turn is complete |
| `ERROR` | engine to client | v2 | something went wrong; the turn is over |
| `MEDIA_OK`, `MEDIA_ERR` | engine to client | v3 | attachment accepted or refused |
| `HW` | engine to client | v3 | hardware the engine runs on |
| `PREFILL` | engine to client | v3 | prompt processing, once per turn |
| `EXPAND` | engine to client | v3 | a tensor being expanded |
| `DTC` | engine to client | v3 | a tensor in dense tensor compute |
| `HS` | engine to client | v3 | Tensor Handshake counters |
| `MEM` | engine to client | v3 | memory |
| `MOE` | engine to client | v3 | experts routed for the current token |
| `STAT` | engine to client | v3 | any other statistic |


Line-based UTF-8 text protocol between the GUI (client, Python) and the
engine (server, pure C). The backend is always C; the Python front end
only renders and relays. Transport: the engine's stdin/stdout when spawned
locally, or a TCP socket when the engine runs with --serve PORT. The
socket listens on 127.0.0.1 unless --serve-any is also given; there is
no authentication, so binding every interface prints a warning.

## Client to engine

    PROMPT <text>\n        one user turn; text is a single line (newlines
                            escaped as \n by the sender)
    STOP\n                  end the current generation early
    RESET\n                 clear the conversation state
    QUIT\n                  end the session

## Engine to client

    READY nemo-engine 2 model=<name> weights=<raw|packed> layers=<n>
         cores=<n> threads=<n> mode=<name> simd=<0|1>\n
                            cores is the processor count the engine
                            queried from the machine at startup;
                            threads is the lane count actually in use;
                            mode is the compute mode (below); simd is
                            1 only when the SIMD path was requested
                            (--simd, or simd = true in littlenemo.ini)
                            and the CPU provides AVX2
    THINK <text>\n          one piece of the reasoning trace (streamed)
    TOK <text>\n            one piece of the visible answer (streamed)
    LAYER <idx> <type>\n    the forward pass entered layer <idx>
                            (type: mamba | mlp | attention)
    TICK mode=<thinking|answering> layer=<idx> next=<idx> active=<n>
         staged=<n> tok=<n> ms_tok=<v> gbs=<v>\n
                            per-token telemetry, sent once per token:
                            mode is what the model is doing with this
                            token; layer/next are the layer being computed
                            and the one after it; active is layers in
                            flight (1 in serialized consumption); staged
                            is the predict-ahead depth (layers whose
                            weights are prefetched and waiting); tok is
                            the running token count; ms_tok is compute
                            time for this token; gbs is the estimated
                            memory throughput (weight bytes touched this
                            token / compute time), an estimate and
                            labeled as one in the GUI.
    STATS tokens=<n> ms=<total> tps=<rate> ms_per_tok=<v> gbs_avg=<v>
          peak_mb=<v>\n     end-of-turn summary
    DONE\n                  the turn is complete (follows STATS)
    ERROR <text>\n          something went wrong; the turn is over

Compute modes: the engine splits each large projection across
worker lanes and prepares the next layer while the current one
computes (preparation concurrent, consumption serialized, results
bit-identical to the single-lane path). --mode selects the lane
count from the queried core count: minimum = 1 lane, medium = half
the cores, max = all cores but one, ultra = every core. --threads N
overrides the mode with an exact lane count (the READY mode then
reads "custom"). An unknown mode is a startup error. --max-new N
caps generated tokens per turn; --max-pos N caps the context.
--simd opts into AVX2 projection kernels: same products in a
fixed blocked accumulation order, deterministic, but possibly
different from the scalar path in the last bits. The scalar path
is the golden reference and stays the default.

Startup validation: READY is printed only after the checkpoint has
passed whole-geometry validation (every tensor the model will fetch,
present and correctly shaped and typed; see nemo_validate_model). A
checkpoint that fails validation exits the engine with the offending
tensor named on stderr, before READY.

State coherence: a forward failure never leaves partially committed
state behind. The engine resets the model state before the next
PROMPT and the ERROR line says so ("...; state reset"). The next
turn always starts a fresh conversation.

Text pieces carry escapes: \\ for backslash, \n for newline, \t for tab.
THINK, TOK, LAYER, and TICK are emitted as generation happens, never
buffered to the end of the turn: the conversation is live at whatever
rate the hardware sustains, and every number is measured, not rounded up.

Thinking split: the engine separates the reasoning trace from the answer
using the model's own delimiters from its chat template. If a model emits
no delimiters, everything is TOK and mode stays answering.

## v3 extensions (backward compatible, 2026-11-26)

Added for the SuperNemo chat GUI: image/video attachments and a full
telemetry feed for the stats panel. Everything here is optional and
additive. A v2 engine never sends these lines and the GUI works
without them, showing a dash or a GUI-measured value (labeled "gui")
instead. A v2 client treats them as unknown lines and ignores them.
The client sends MEDIA only to an engine whose READY advertised it.

Values on the new key=value lines are percent-encoded (RFC 3986:
every byte outside A-Z a-z 0-9 - . _ ~ is %XX, UTF-8), so names,
paths and device strings may contain spaces, '=' and non-ASCII
text. Numbers are plain. Lines longer than 256 KiB are ignored by
the client.

### READY additions

    media=<image,video|image|video>   attachment types the engine
                                      accepts; absent = text only
    accel=<cpu|cuda|opencl-igpu|opencl-nv>  active compute lane
    moe=<n>                           experts per MoE layer, if any
    fixture=1                         a test fixture, not a model; the
                                      GUI shows a TEST FIXTURE badge

### Client to engine

    MEDIA id=<n> type=<image|video> mime=<mime> bytes=<n>
          [w=<px> h=<px>] [frames=<n>] [dur_ms=<n>]
          [sent_frames=<n>] name=<pct> path=<pct>\n
                            announces one attachment for the next
                            PROMPT. Sent immediately before that
                            PROMPT; ids are unique per session. The
                            path is absolute on the GUI's machine, so
                            a remote engine needs a shared path.
    FRAME id=<n> idx=<i> ts_ms=<t> path=<pct>\n
                            one extracted video frame (PNG) for
                            MEDIA id; exactly sent_frames of them
                            follow their MEDIA line

Client-side bounds, checked before anything is sent: at most 8
attachments per turn; images up to 64 MiB, 16384 px per side and
100 MP; videos up to 4 GiB and at most 32 frames; extension and
leading bytes must agree (PNG, JPEG, GIF, BMP, WebP; MP4/M4V/MOV,
WebM/MKV, AVI, animated GIF/WebP). The engine must validate again:
the file exists, its size equals bytes=, type is known, id is not a
duplicate, and the count and frame limits hold. RESET drops pending
attachments.

### Engine to client

    MEDIA_OK id=<n> [type=<t>] [tokens=<n>]\n
                            the attachment was accepted and costs
                            <n> context tokens
    MEDIA_ERR id=<n> reason=<pct>\n
                            the attachment was refused; the turn
                            goes on without it
    HW key=value...\n       hardware the engine runs on: cpu cores
                            threads ram_mb gpu vram_mb driver igpu
                            accel os (any subset); overrides what the
                            GUI probed from its own machine
    PREFILL tokens=<n> ms=<v> tps=<v>\n
                            prompt processing, sent once per turn
                            before the first THINK/TOK
    EXPAND layer=<i> tensor=<pct> [rows= cols= bytes= lane=]
           state=<begin|end>\n
                            a tensor being expanded (dequantized)
    DTC layer=<i> tensor=<pct> [rows= cols= k= block= active= lane=]
        state=<begin|end>\n
                            a tensor in dense tensor compute: k is the
                            prototype dictionary size, block the block
                            length, active the active prototypes
    HS pass=<n> ordinal=<n> hits=<n> misses=<n> stall_ms=<v>
       dtc=<n> dtc_ms=<v> [staged=<n>] [more counters]\n
                            Tensor Handshake counters, cumulative per
                            session. DTC requests are counted in dtc,
                            not as prefetch misses, and DTC compute
                            time in dtc_ms, not as stall time
    MEM ram_mb=<v> peak_mb=<v> [vram_mb=<v>] [vram_peak_mb=<v>]\n
    MOE layer=<i> experts=<pct list> [k=<n>] [of=<n>]\n
                            the experts routed for the current token
    STAT key=value...\n     any other statistic; shown as-is under
                            "Other engine stats"

STATS may carry more keys: prompt_tokens, prompt_ms, prompt_tps,
ttft_ms, gen_tps, think_tokens, answer_tokens, stopped=<0|1>, and
others, all shown in the stats panel. READY's version field reads 3
for an engine that speaks these extensions; clients must not rely on
it, only on the keys and lines themselves.
