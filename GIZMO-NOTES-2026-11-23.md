# NanoQuatTwo: Inference Engine Research Notes & Optimization Proposals

Source: Gizmo (ChatGPT), outside technical consultant, relayed by JD
2026-11-23. Received in part: sections 1 through 3 (the paste ends
mid-formula in section 3). Remainder pending; append when it arrives.

---

## 1. Project objective

The central objective is to develop a compressed neural-network weight representation that can be stored at a substantially smaller size, reconstructed or expanded as needed, and streamed into computation without requiring the entire model to reside in GPU memory.

The intended architecture resembles virtual texture and geometry streaming in modern game engines.

A game engine can maintain a world much larger than available GPU memory by retrieving the textures and geometry needed for the current working set. The proposed NanoQuatTwo architecture applies a similar principle to model weights.

The intended pipeline:

1. Store the complete model in a compressed representation.
2. Divide the representation into addressable books or weight blocks.
3. Predict which blocks will be needed next.
4. Retrieve them from persistent storage.
5. Stage them in system memory.
6. Reconstruct or expand the required representations.
7. Transfer or expose the active working set to computation.
8. Recycle buffers and continue streaming.

**The objective is not merely to compress a model. It is to make compressed model weights practical to retrieve, reconstruct, and consume continuously during inference.**

The principal hardware target under investigation is a laptop with approximately 4 GB of GPU VRAM and a 12-thread CPU.

An ambitious target is to take a model originally around 8 GB and produce a compressed artifact below 100 MB. That target represents at least approximately 80:1 size reduction, but preservation of useful model behavior remains to be demonstrated.

---

## 2. Experimental origin and progression

The project began with a simple question:

> How much information can we pack into one quad word, and can we retrieve it reliably?

The experimental progression:

1. Test the fundamental quad-word representation.
2. Increase the amount of encoded information.
3. Test retrieval and reconstruction reliability.
4. Create examples using recognizable characters.
5. Explore a taught-plus-learned codebook.
6. Increase the number of codebook entries.
7. Investigate actual neural-network weight quantization.
8. Develop an inference engine capable of consuming the representation.
9. Investigate predictive memory prefetching and streaming execution.

The progression matters because the project evolved from a controlled representation experiment into a model-weight storage and execution architecture.

The research should preserve this experimental history.

---

## 3. Core representation: NanoQuatTwo

The fundamental concept is a dictionary-based representation of weight blocks.

Instead of storing every individual weight directly:

- Divide weights into blocks.
- Maintain a codebook containing representative blocks.
- Represent each original block with an index into the codebook.
- Pack the indices into compact quad-word storage.
- Retrieve the selected prototype when reconstruction or computation requires it.

Conceptually:

`Original tensor → Blocks → Codebook matching → Packed indices`

Reconstruction:

`Packed index → Codebook lookup → Reconstructed block`

The important distinction is between the compressed index stream and the complete compressed artifact.

The codebook must always be included in storage accounting.

### Effective bits per weight

Let:

- S = weights per block.
- K = number of codebook entries.
- B = number of blocks.
- b = bits used to store each codebook weight.
- q = ceil(log2 K) = index bits per block.

The index-stream cost is:

M_index = Bq

The codebook cost is:

M_book = KSb

Total storage, excluding metadata:

M_total = Bq + KSb

[Paste ends here, mid-document.]
