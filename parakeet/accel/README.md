# accel/ (reserved)

This folder is reserved for an optional CUDA lane for the parakeet-c encoder GEMMs (bf16 tensor cores on the RTX 3050 Ti).
It is not built: CPU-only parakeet-c already runs at about 0.06 RTF, and the 4 GB card is needed for the Nemo chat weights.
The CPU path stays the default and the reference.
