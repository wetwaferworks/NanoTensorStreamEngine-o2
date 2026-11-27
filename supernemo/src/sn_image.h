/* sn_image.h - image attachments for the SuperNemo vision tower (pure C99).
 *
 * Mirrors image_processing_nemotron_h_omni.py: choose a patch grid by the checkpoint's dynamic-resolution rule,
 * resize with antialiased bicubic (torch.nn.functional.interpolate, align_corners=False, a = -0.5), scale by 1/255,
 * normalise with the checkpoint's mean/std, planar [3][H][W] float32. Decoding is stb_image (PNG, JPEG, BMP, GIF). */
#ifndef SN_IMAGE_H
#define SN_IMAGE_H

/* Decode a file to 8-bit RGB (malloc'ed, caller frees). 0 on success; on failure a short reason is in err. */
int sn_image_load(const char *path, unsigned char **rgb, int *w, int *h, char *err, int errlen);

/* The processor's _compute_target_patches: tile (wp, hp) in patches for a w x h image.
 * budget = patches allowed (the reference uses 13312; a CPU run may pass fewer), min_patches 1024,
 * shuffle = pixel shuffle factor (2). */
void sn_image_plan(int w, int h, int patch, int shuffle, int min_patches, int budget, int *wp, int *hp);

/* Resize to (wp*patch) x (hp*patch) and normalise; returns planar [3][H][W] floats (malloc'ed) or NULL. */
float *sn_image_prepare(const unsigned char *rgb, int w, int h, int wp, int hp, int patch, const float mean[3], const float stdv[3]);

/* Antialiased bicubic of one float plane (exposed for tests). */
void sn_image_resize_plane(const float *src, int sw, int sh, float *dst, int dw, int dh);

#endif
