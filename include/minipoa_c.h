/*
 * A C-callable multiple-sequence-alignment entry point for minipoa.
 *
 * minipoa's own headers are C++ (std::vector, std::string, constructors, overloads, default
 * arguments), so they cannot be included from C.  This header is the only one a C caller needs:
 * opaque handle, plain scalars, no STL, no klib, no <zlib.h>.
 *
 * The shape deliberately mirrors abPOA's, because the caller this exists for -- cactus's bar
 * phase -- already speaks that shape.  In particular the scoring setters take abPOA's
 * conventions (gap penalties POSITIVE, matrix row-major over A,C,G,T,N) and convert internally;
 * minipoa maximises, so its own gap fields are negative.
 *
 * Released under the MIT license, see LICENSE.
 */
#ifndef MINIPOA_C_H
#define MINIPOA_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Value used for a gap in the returned MSA.  Matches abPOA, and minipoa's own nt256_table[5]. */
#define MINIPOA_GAP 5

/* Alphabet size; the substitution matrix is MINIPOA_ALPHA x MINIPOA_ALPHA over A,C,G,T,N. */
#define MINIPOA_ALPHA 5

typedef struct minipoa_para_s minipoa_para_t;

/* Allocate a parameter block with minipoa's defaults.  Free with minipoa_free_para(). */
minipoa_para_t *minipoa_init_para(void);
void minipoa_free_para(minipoa_para_t *p);

/*
 * mat25 is 25 ints, row-major, indexed [graph_base * 5 + query_base] over A,C,G,T,N -- exactly
 * abPOA's abpt->mat layout and encoding, so the same config string feeds both aligners.
 * Scores are signed as given (matches positive, mismatches negative).
 */
void minipoa_set_score_matrix(minipoa_para_t *p, const int *mat25);

/*
 * Gap penalties in abPOA's convention: POSITIVE numbers, cost of a length-L gap being
 * gap_open + L * gap_ext.
 */
void minipoa_set_gap(minipoa_para_t *p, int gap_open, int gap_ext);
/*
 * An optional second gap piece, also POSITIVE, making the gap cost convex exactly as abPOA's is:
 * a length-L gap costs min(gap_open + L * gap_ext, gap_open2 + L * gap_ext2).  It adds two dp
 * matrices, so costs about half as much again in memory and time.  Both 0 (the default) turns
 * it off.
 */
void minipoa_set_gap2(minipoa_para_t *p, int gap_open2, int gap_ext2);

/*
 * Band half-width, expressed as abPOA does it: band = band_constant + band_fraction * qlen.
 * minipoa takes an integer divisor instead of a fraction, so band_fraction is converted to
 * f = round(1 / band_fraction) and only fractions of the form 1/n are exactly representable.
 * band_fraction <= 0 turns banding off entirely (full DP).
 */
void minipoa_set_band(minipoa_para_t *p, int band_constant, double band_fraction);

/*
 * Adaptive banding.  Off by default, and worth leaving off: the adaptive path sizes its scratch
 * for the FULL matrix before narrowing the band, so it is a compute optimisation, not a memory
 * one.
 */
void minipoa_set_adaptive_band(minipoa_para_t *p, int on);

/*
 * Minimizer seeding.  When on, each sequence is anchored against the graph's consensus path and
 * aligned one bounded interval at a time -- this is what keeps minipoa's memory flat on long
 * inputs.  It is also the risky knob: a false-positive anchor becomes a forced match and cannot
 * be recovered from.  anchor_window of 0 keeps minipoa's own default spacing.
 */
void minipoa_set_seeding(minipoa_para_t *p, int enable, int k, int w, int anchor_window);

/* Guide-tree ordering.  O(N^2) in time and memory; off by default. */
void minipoa_set_progressive(minipoa_para_t *p, int on);

/*
 * Align n_seqs sequences and return the row-column MSA.
 *
 * seqs[i] holds seq_lens[i] bytes, values 0..4 for A,C,G,T,N (the same encoding abPOA takes).
 * On success returns 0, sets *col_no_out, and sets *msa_out to n_seqs rows of *col_no_out bytes
 * in INPUT ORDER, each cell either a 0..4 base or MINIPOA_GAP.  Each row is a separate
 * allocation: free every row, then the row array.
 *
 * On failure returns non-zero, allocates nothing, and leaves *msa_out untouched;
 * minipoa_last_error() describes what happened.  No exception escapes this function.
 */
int minipoa_msa(const minipoa_para_t *p, int n_seqs, const int *seq_lens,
                uint8_t *const *seqs, uint8_t ***msa_out, int *col_no_out);

/* Message for the most recent failure on this thread.  Never NULL. */
const char *minipoa_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
