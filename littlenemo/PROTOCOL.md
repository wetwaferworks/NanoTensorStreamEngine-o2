# Little Nemo engine protocol v2

Line-based UTF-8 text protocol between the GUI (client, Python) and the
engine (server, pure C). The backend is always C; the Python front end
only renders and relays. Transport: the engine's stdin/stdout when spawned
locally, or a TCP socket when the engine runs with --serve PORT.

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
