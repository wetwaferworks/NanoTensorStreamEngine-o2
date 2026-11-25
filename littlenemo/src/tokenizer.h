/* tokenizer.h - Little Nemo byte-level BPE tokenizer, pure C.
 * Reads a Hugging Face tokenizer.json (the Nemotron checkpoint's):
 * model.vocab, model.merges (string "A B" or pair-array form), and
 * added_tokens for the specials. Encoding is the standard greedy
 * lowest-rank merge over byte-mapped symbols (GPT-2 byte-to-unicode
 * mapping); decoding reverses it. Added tokens appearing in the
 * input are extracted and emitted as their own ids before BPE,
 * matching the reference tokenizer (Sol M1); specials are also
 * looked up by content ("<unk>" and kin) for the template layer.
 */
#ifndef NEMO_TOKENIZER_H
#define NEMO_TOKENIZER_H

typedef struct nemo_tokenizer nemo_tokenizer;

/* Load from a tokenizer.json path. NULL on any failure. */
nemo_tokenizer *nemo_tokenizer_load(const char *path);
void nemo_tokenizer_free(nemo_tokenizer *tk);

int nemo_tokenizer_vocab_size(const nemo_tokenizer *tk);

/* Id of an added (special) token by its content, or -1. */
int nemo_tokenizer_special(const nemo_tokenizer *tk, const char *content);

/* Encode UTF-8 text to token ids. Returns the count, or -1 on error
 * (including max_ids too small or a byte with no token). */
int nemo_tokenizer_encode(const nemo_tokenizer *tk, const char *utf8,
                          int *ids_out, int max_ids);

/* Decode ids to a malloc'd UTF-8 string (caller frees). Special
 * tokens are emitted literally unless skip_specials is set. */
char *nemo_tokenizer_decode(const nemo_tokenizer *tk, const int *ids,
                            int n, int skip_specials);

#endif
