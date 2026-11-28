# How NTSE compares with llama.cpp

NTSE is similar to [llama.cpp](https://github.com/ggml-org/llama.cpp): a local inference engine written in C that runs quantized language models on ordinary computers, without sending the model or the conversation off the machine.

A few things are different. The runtime is pure C99, with no other language in the engine. There are no mutexes in shipped code. Results are hash-identical across thread counts. It runs NVIDIA's hybrid Mamba models, and it streams mixture-of-experts weights through a 4 GB card so a 120-billion-parameter SuperNemo model can run there.

That is the whole comparison. We have not run llama.cpp side by side on this card, so we do not claim to be faster or slower.
