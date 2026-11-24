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

    READY nemo-engine 2 model=<name> weights=<raw|packed> layers=<n>\n
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

Text pieces carry escapes: \\ for backslash, \n for newline, \t for tab.
THINK, TOK, LAYER, and TICK are emitted as generation happens, never
buffered to the end of the turn: the conversation is live at whatever
rate the hardware sustains, and every number is measured, not rounded up.

Thinking split: the engine separates the reasoning trace from the answer
using the model's own delimiters from its chat template. If a model emits
no delimiters, everything is TOK and mode stays answering.
