/*
 * C-callable wrapper around minipoa's C++ alignment API.  See include/minipoa_c.h.
 *
 * Released under the MIT license, see LICENSE.
 */
#include "minipoa_c.h"

#include "align.h"
#include "file_io.h"
#include "graph.h"
#include "minimizer.h"
#include "parameter.h"
#include "sequence.h"

#include <cmath>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <vector>

struct minipoa_para_s {
  para_t para;
};

namespace {

/*
 * initPara() fills char26_table/char256_table, which are process-wide mutable globals read on
 * every hot path -- including graph::init(), before a single base is aligned.  Every nucleotide
 * caller writes the same 512 bytes, so the race is benign in practice, but it is still a race and
 * a thread sanitiser will say so.  Do it once.
 *
 * The cost of getting this wrong is not a crash: the tables are zero-initialised, so skipping
 * initPara() maps every base to 'A' and produces silently wrong alignments.
 */
void init_alphabet_tables_once() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    para_t probe = para_t(); // value-init: para_t has no constructor and no member initialisers
    probe.match = 1;
    probe.mismatch = -1;
    probe.gap_open1 = -1;
    probe.gap_ext1 = -1;
    initPara(&probe); // mat_fp empty -> nucleotide tables
  });
}

std::string &last_error_slot() {
  static thread_local std::string msg("no error");
  return msg;
}

void set_error(const char *what) { last_error_slot().assign(what ? what : "unknown error"); }

} // namespace

extern "C" {

minipoa_para_t *minipoa_init_para(void) {
  try {
    init_alphabet_tables_once();
    minipoa_para_t *h = new minipoa_para_t();
    para_t &p = h->para;

    /*
     * initPara() sets only mm_filter_ratio, m, mat, the sign normalisation and the char tables.
     * Everything else in para_t is left at whatever the struct was constructed with -- and para_t
     * has no constructor and no default member initialisers, so value-initialise first and then
     * set every field that matters explicitly.  m == 0 in particular is not survivable:
     * graph::init() builds node_t(id, base, m), whose constructor does
     * aligned_node.resize(m, -1); aligned_node[base] = id -- an out-of-bounds write.
     */
    p.match = 2;
    p.mismatch = -4;
    p.gap_open1 = -4;
    p.gap_ext1 = -2;
    p.mat_fp.clear();
    p.inc_fp.clear();
    initPara(&p); // -> m = 5, default mat, char tables, signs normalised

    p.b = 100;
    p.f = 40;
    p.ab_band = false;
    p.k = 19;
    p.mm_w = 10;
    p.poa_w = 0;
    p.enable_seeding = false;
    p.progressive_poa = false;
    p.isRNA = false;
    p.verbose = 0;

    // bw is hardcoded in minipoa's own main() and is required by dp_chaining(); nothing in
    // initPara() sets it, and 0 would make the chaining band degenerate.
    p.bw = 1000;

    // The caller owns parallelism.  alignment() opens its own omp region when thread > 1 and
    // indexes mpool[omp_get_thread_num()], which would run off the single-element pool below.
    p.thread = 1;

    // NOT an output-format selector despite the name.  graph::add_path() only accumulates
    // path_node_ids when this is non-zero, and those paths are the entire input to
    // get_rc_msa().  Leave it at 0 and every row comes back all gaps, successfully.
    p.result = 1;

    return h;
  } catch (const std::exception &e) {
    set_error(e.what());
    return 0;
  } catch (...) {
    set_error("minipoa: unknown error allocating parameters");
    return 0;
  }
}

void minipoa_free_para(minipoa_para_t *p) { delete p; }

void minipoa_set_score_matrix(minipoa_para_t *p, const int *mat25) {
  if (!p || !mat25) return;
  // Safe to overwrite after initPara(): nothing re-derives mat, and match/mismatch are read only
  // when the default matrix is synthesised.  There is no max_mat/min_mis to keep in step, unlike
  // abPOA.
  p->para.m = MINIPOA_ALPHA;
  p->para.mat.assign(mat25, mat25 + MINIPOA_ALPHA * MINIPOA_ALPHA);
}

void minipoa_set_gap(minipoa_para_t *p, int gap_open, int gap_ext) {
  if (!p) return;
  // minipoa maximises and adds these directly, so penalties are negative here.
  p->para.gap_open1 = -(gap_open < 0 ? -gap_open : gap_open);
  p->para.gap_ext1 = -(gap_ext < 0 ? -gap_ext : gap_ext);
}

void minipoa_set_band(minipoa_para_t *p, int band_constant, double band_fraction) {
  if (!p) return;
  p->para.b = band_constant;
  if (band_fraction > 0.0) {
    int f = (int)(1.0 / band_fraction + 0.5);
    p->para.f = f > 0 ? f : 1;
  } else {
    p->para.f = 0; // full DP
  }
}

void minipoa_set_adaptive_band(minipoa_para_t *p, int on) {
  if (p) p->para.ab_band = on != 0;
}

void minipoa_set_seeding(minipoa_para_t *p, int enable, int k, int w, int anchor_window) {
  if (!p) return;
  p->para.enable_seeding = enable != 0;
  if (k > 0) p->para.k = k;
  if (w > 0) p->para.mm_w = w;
  p->para.poa_w = anchor_window > 0 ? anchor_window : 0;
}

void minipoa_set_progressive(minipoa_para_t *p, int on) {
  if (p) p->para.progressive_poa = on != 0;
}

int minipoa_msa(const minipoa_para_t *p, int n_seqs, const int *seq_lens,
                uint8_t *const *seqs, uint8_t ***msa_out, int *col_no_out) {
  if (!p || !seq_lens || !seqs || !msa_out || !col_no_out) {
    set_error("minipoa: null argument");
    return 1;
  }
  if (n_seqs <= 0) {
    set_error("minipoa: no sequences");
    return 1;
  }
  try {
    init_alphabet_tables_once();

    // Work on a copy: alignment()/graph take a non-const para_t, and a caller's handle may be
    // shared across threads.  para_t deep-copies its vector and strings.
    para_t para = p->para;

    /*
     * minimizer_t is built from the whole sequence set up front, so it needs them as seq_t.
     * The 0..4 encoding passes through untouched: nt4_table[0..4] == {0,1,2,3,4}, and
     * alignment() re-encodes through char26_table, which is idempotent on those values.
     */
    std::vector<seq_t> seq_v((size_t)n_seqs);
    for (int i = 0; i < n_seqs; i++) {
      const int len = seq_lens[i];
      if (len < 0) {
        set_error("minipoa: negative sequence length");
        return 1;
      }
      seq_v[(size_t)i].seq.assign((const char *)seqs[i], (size_t)len);
    }

    // Braces matter: graph has no constructor, so plain `graph dag;` leaves the POD members
    // indeterminate, and alignment()'s seeding path reads is_topsorted before anything sets it.
    graph dag = graph();
    // Required.  This is what creates src (node 0) and sink (node 1); without it the first base
    // becomes node 0, add_adj(0, 0) is swallowed as a self-loop, and the graph has no terminals.
    dag.init(&para);

    // Needed even with seeding off: alignment() takes it and reads mm->seqs_size.  With
    // progressive_poa false the constructor is cheap and leaves km null, so kalloc falls through
    // to plain malloc/free.
    minimizer_t mm(&para, seq_v);
    if (para.progressive_poa) mm.get_guide_tree(&para);

    // One scratch buffer, constructed and destroyed per call.  It grows by doubling and never
    // shrinks, so a long-lived one would pin the largest window's high-water mark for the
    // process.  Passing null instead would leak it if the DP throws.
    aligned_buff_t pool;

    for (size_t step = 0; step < seq_v.size(); step++) {
      const int rid = mm.ord.empty() ? (int)step : mm.ord[step];
      std::vector<res_t> res = alignment(&para, &dag, &mm, rid,
                                         seq_v[(size_t)rid].seq.c_str(),
                                         (int)seq_v[(size_t)rid].seq.size(), &pool);
      dag.add_path(&para, rid, res, /*writer=*/0, /*sink_id=*/1);
      dag.topsort(&para, 0);
    }

    int cols = 0;
    uint8_t **msa = dag.get_rc_msa(&para, n_seqs, &cols, (uint8_t)MINIPOA_GAP);
    *msa_out = msa;
    *col_no_out = cols;
    return 0;
  } catch (const std::exception &e) {
    // Nothing may cross the C boundary: cactus's callers are C frames with no unwind tables.
    set_error(e.what());
    return 1;
  } catch (...) {
    set_error("minipoa: unknown error during alignment");
    return 1;
  }
}

const char *minipoa_last_error(void) { return last_error_slot().c_str(); }

} // extern "C"
