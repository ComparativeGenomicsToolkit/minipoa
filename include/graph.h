#ifndef GRAPH_H
#define GRAPH_H
#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include "node.h"
#include "result.h"
#include "sequence.h"
#include "parameter.h"
#include "file_io.h"

class PathWriter;

struct graph {
  // 成员声明
  std::vector<node_t> node; //index is id
  // std::vector<int> node_h;  //index is ord[id] 
  std::vector<int> rank; // rank_to_node_id
  std::vector<int> hmin, hmax, tmin, tmax;  // index is rank  haid tail
  std::vector<int> hlen, tlen;  // index is rank  haid tail
  void init(para_t* para);
  void init(para_t *para, int seq_id, const std::string &str, PathWriter *writer);
  int add_node(para_t* para, char base);
  void add_adj(int from, int to);
  void add_path(const para_t *para, int seq_id, const std::vector<res_t> &res, PathWriter *writer, int sink_id = -1);
  /*
   * Row-column MSA as data rather than as FASTA on stdout.
   *
   * Rows are indexed by the ORIGINAL sequence id, so the caller gets them back in input order
   * even when a guide tree reorders the alignment.  Bases are the internal 0..4 encoding, not
   * ASCII; `gap_val` fills everything else.  One malloc per row plus one for the row array, so
   * the caller frees with free(row) per row then free(rows).
   *
   * Requires the in-memory path route, i.e. add_path() called with writer == nullptr.
   */
  uint8_t **get_rc_msa(para_t *para, int n_seq, int *column_no, uint8_t gap_val);
  void topsort(const para_t* para, int op);
  void output_rc_msa(para_t* para, const std::vector<int>& rid_to_ord, const std::vector<seq_t>& seqs);
  void output_consensus(bool needCoverages = false);
  void build_consensus(bool needCoverages = false);
  void output_gfa(const std::vector<int>& rid_to_ord, const std::vector<seq_t>& seqs);
  std::vector<int> calculateR() const;
  bool is_topsorted; //
  std::string cons;
  std::vector<int> cons_pos_to_id;
  std::vector<int> coverages;
  std::string tmp_path;
  /*
   * Per-sequence node paths, kept in memory as (original seq_id, node ids).
   *
   * The CLI journals these to tmp_path instead, via PathWriter.  That file is named only by pid
   * and its writer unlinks before opening, so two graphs alive in one process clobber each
   * other -- fine for a one-shot binary, fatal for a library called from several threads.
   * add_path() fills this vector instead whenever writer == nullptr.
   */
  std::vector<std::pair<int, std::vector<int> > > paths;
};
// std::vector<sequence> readFile(const char* path);
#endif