#ifndef SIMDPRIV_H
#define SIMDPRIV_H
/*
 * On x86 this is the real intrinsics header.  Elsewhere -- cactus builds for aarch64 too -- simde
 * maps the same intrinsics onto NEON, which is exactly how abPOA handles the same problem.
 * Define USE_SIMDE and SIMDE_ENABLE_NATIVE_ALIASES and put simde on the include path; the code
 * below is then unchanged.
 */
#if defined(USE_SIMDE)
#include <simde/x86/sse4.1.h>
#include <simde/x86/avx2.h>
#if defined(ENABLE_AVX512)
#include <simde/x86/avx512.h>
#endif
#else
#include <immintrin.h>
#endif
#include <cstddef>
#include <cstdint>

// ============================================================
//  SIMD backend selection (controlled by CMake)
// ============================================================

// ============================================================
//  SIMD register & width
// ============================================================

#if defined(ENABLE_AVX512)
using simd_reg = __m512i;
constexpr size_t SIMD_BYTES = 64;
#elif defined(ENABLE_AVX2)
using simd_reg = __m256i;
constexpr size_t SIMD_BYTES = 32;
#else
using simd_reg = __m128i;
constexpr size_t SIMD_BYTES = 16;
#endif

constexpr int simd_width = SIMD_BYTES / sizeof(int);

// ============================================================
//  Load / Store
// ============================================================

inline simd_reg simd_load(const int* p) {
#if defined(ENABLE_AVX512)
  return _mm512_load_si512(p);
#elif defined(ENABLE_AVX2)
  return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
#else
  return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
#endif
}

inline simd_reg simd_loadu(const int* p) {
#if defined(ENABLE_AVX512)
  return _mm512_loadu_si512(p);
#elif defined(ENABLE_AVX2)
  return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
#else
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
#endif
}

inline void simd_store(int* p, simd_reg v) {
#if defined(ENABLE_AVX512)
  _mm512_store_si512(p, v);
#elif defined(ENABLE_AVX2)
  _mm256_store_si256(reinterpret_cast<__m256i*>(p), v);
#else
  _mm_store_si128(reinterpret_cast<__m128i*>(p), v);
#endif
}

// ============================================================
//  Set
// ============================================================

inline simd_reg simd_set1(int x) {
#if defined(ENABLE_AVX512)
  return _mm512_set1_epi32(x);
#elif defined(ENABLE_AVX2)
  return _mm256_set1_epi32(x);
#else
  return _mm_set1_epi32(x);
#endif
}

// ============================================================
//  Arithmetic
// ============================================================

inline simd_reg simd_add(simd_reg a, simd_reg b) {
#if defined(ENABLE_AVX512)
  return _mm512_add_epi32(a, b);
#elif defined(ENABLE_AVX2)
  return _mm256_add_epi32(a, b);
#else
  return _mm_add_epi32(a, b);
#endif
}

inline simd_reg simd_sub(simd_reg a, simd_reg b) {
#if defined(ENABLE_AVX512)
  return _mm512_sub_epi32(a, b);
#elif defined(ENABLE_AVX2)
  return _mm256_sub_epi32(a, b);
#else
  return _mm_sub_epi32(a, b);
#endif
}

inline simd_reg simd_max(simd_reg a, simd_reg b) {
#if defined(ENABLE_AVX512)
  return _mm512_max_epi32(a, b);
#elif defined(ENABLE_AVX2)
  return _mm256_max_epi32(a, b);
#elif defined(__SSE4_1__)
  return _mm_max_epi32(a, b);
#else
  // _mm_max_epi32 is SSE4.1, not SSE2 -- it is the single intrinsic that forced the ENABLE_SSE2
  // backend to be compiled with -msse4.1 despite its name.  Emulate it so a real -msse2 build
  // works; simd_argmax already does the same blend this way.
  simd_reg mask = _mm_cmpgt_epi32(a, b);
  return _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b));
#endif
}

// ============================================================
//  DP helper
//  lane 0 = prev
//  lane i = p[i-1]
// ============================================================

inline simd_reg simd_set_prev_and_load(int prev, const int* p) {
#if defined(ENABLE_AVX512)
  return _mm512_setr_epi32(
      prev,
      p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
      p[8], p[9], p[10], p[11], p[12], p[13], p[14]
  );
#elif defined(ENABLE_AVX2)
  return _mm256_setr_epi32(
      prev,
      p[0], p[1], p[2], p[3], p[4], p[5], p[6]
  );
#else
  return _mm_setr_epi32(
      prev,
      p[0], p[1], p[2]
  );
#endif
}

// ============================================================
//  16-bit storage
//  simd_width unsigned 16-bit values, widened to / narrowed from the 32-bit lanes of a simd_reg.
//  p must be aligned to simd_width * 2 bytes.
// ============================================================

inline simd_reg simd_load_u16(const uint16_t* p) {
#if defined(ENABLE_AVX512)
  return _mm512_cvtepu16_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(p)));
#elif defined(ENABLE_AVX2)
  return _mm256_cvtepu16_epi32(_mm_load_si128(reinterpret_cast<const __m128i*>(p)));
#else
  return _mm_unpacklo_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)), _mm_setzero_si128());
#endif
}

// simd_width 16-bit values: half a simd_reg (a quarter of one on the 128-bit path)
#if defined(ENABLE_AVX512)
using simd_half = __m256i;
#else
using simd_half = __m128i;
#endif

// v must be non-negative; lanes above 65535 come out as 65535
inline simd_half simd_pack_u16_sat(simd_reg v) {
#if defined(ENABLE_AVX512)
  return _mm512_cvtusepi32_epi16(v);
#elif defined(ENABLE_AVX2)
  // packus works within each 128-bit half, so gather the two halves' results into the low one
  return _mm256_castsi256_si128(_mm256_permute4x64_epi64(_mm256_packus_epi32(v, v), 0x08));
#else
  // packus_epi32 is SSE4.1: shift into signed range, saturate with the SSE2 signed pack, shift back
  __m128i biased = _mm_sub_epi32(v, _mm_set1_epi32(32768));
  return _mm_xor_si128(_mm_packs_epi32(biased, biased), _mm_set1_epi16((short)0x8000));
#endif
}

inline void simd_store_half(uint16_t* p, simd_half h) {
#if defined(ENABLE_AVX512)
  _mm256_store_si256(reinterpret_cast<__m256i*>(p), h);
#elif defined(ENABLE_AVX2)
  _mm_store_si128(reinterpret_cast<__m128i*>(p), h);
#else
  _mm_storel_epi64(reinterpret_cast<__m128i*>(p), h);
#endif
}

inline void simd_store_u16_sat(uint16_t* p, simd_reg v) { simd_store_half(p, simd_pack_u16_sat(v)); }

// ============================================================
//  Non-temporal (streaming) stores: to memory without first reading the line into cache.  For
//  data written once and not read again soon.  simd_stream_fence() orders them before later
//  stores; the 64-bit-lane path has no 8-byte streaming store and stores normally.
// ============================================================

inline void simd_stream(int* p, simd_reg v) {
#if defined(ENABLE_AVX512)
  _mm512_stream_si512(reinterpret_cast<__m512i*>(p), v);
#elif defined(ENABLE_AVX2)
  _mm256_stream_si256(reinterpret_cast<__m256i*>(p), v);
#else
  _mm_stream_si128(reinterpret_cast<__m128i*>(p), v);
#endif
}

inline void simd_stream_half(uint16_t* p, simd_half h) {
#if defined(ENABLE_AVX512)
  _mm256_stream_si256(reinterpret_cast<__m256i*>(p), h);
#elif defined(ENABLE_AVX2)
  _mm_stream_si128(reinterpret_cast<__m128i*>(p), h);
#else
  _mm_storel_epi64(reinterpret_cast<__m128i*>(p), h);
#endif
}

inline void simd_stream_fence() { _mm_sfence(); }

// ============================================================
//  Lane shifts, for the horizontal gap scan
//  lane i = v[i - S], lanes below S = fill (fill must hold the same value in every lane)
// ============================================================

template <int S>
inline simd_reg simd_shift_lanes_up(simd_reg v, simd_reg fill) {
#if defined(ENABLE_AVX512)
  return _mm512_alignr_epi32(v, fill, 16 - S);
#elif defined(ENABLE_AVX2)
  // t = [fill low half, v low half]; the per-128-bit alignr then pulls each half's missing low
  // lanes from t.  S == 4 is t itself (alignr by 0).
  __m256i t = _mm256_permute2x128_si256(v, fill, 0x02);
  return _mm256_alignr_epi8(v, t, 16 - 4 * S);
#else
  // Plain SSE2, so the -msse2 build works too.
  return _mm_or_si128(_mm_slli_si128(v, 4 * S), _mm_srli_si128(fill, 16 - 4 * S));
#endif
}

// every lane = the last lane of v
inline simd_reg simd_broadcast_last(simd_reg v) {
#if defined(ENABLE_AVX512)
  return _mm512_permutexvar_epi32(_mm512_set1_epi32(15), v);
#elif defined(ENABLE_AVX2)
  return _mm256_permutevar8x32_epi32(v, _mm256_set1_epi32(7));
#else
  return _mm_shuffle_epi32(v, 0xFF);
#endif
}

/*
 * S[v] = max over u <= v of (x[u] + (v - u) * e), a running max that decays by e (<= 0) per lane,
 * in log2(simd_width) shift/add/max steps.  Ek[s] holds s * e broadcast, for s = 1, 2, 4, 8.
 * fill must be low enough that fill + (simd_width / 2) * e can never beat a real score, and must
 * not overflow when that is added to it.
 */
inline simd_reg simd_decaying_prefix_max(simd_reg x, simd_reg fill, simd_reg E1, simd_reg E2, simd_reg E4, simd_reg E8) {
  x = simd_max(x, simd_add(simd_shift_lanes_up<1>(x, fill), E1));
  x = simd_max(x, simd_add(simd_shift_lanes_up<2>(x, fill), E2));
#if defined(ENABLE_AVX512) || defined(ENABLE_AVX2)
  x = simd_max(x, simd_add(simd_shift_lanes_up<4>(x, fill), E4));
#endif
#if defined(ENABLE_AVX512)
  x = simd_max(x, simd_add(simd_shift_lanes_up<8>(x, fill), E8));
#endif
  (void)E4; (void)E8;
  return x;
}

// ============================================================
//  ArgMax (Find index of maximum value)
// ============================================================

// 辅助函数：初始化 0 到 simd_width-1 的递增下标序列
inline simd_reg simd_set_sequence() {
#if defined(ENABLE_AVX512)
  return _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#elif defined(ENABLE_AVX2)
  return _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
#else
  return _mm_setr_epi32(0, 1, 2, 3);
#endif
}

// 核心函数：向量化寻找最大值的下标
inline int simd_argmax(const int* M, int block_num) {
  if (block_num <= 0) return 0;

  // 初始化：加载第一个块的值，并设定初始对应的下标
  simd_reg max_vals = simd_load(M);
  simd_reg max_idxs = simd_set_sequence();
  
  simd_reg curr_idxs = max_idxs;
  simd_reg idx_inc = simd_set1(simd_width);

  // 遍历后续所有的 SIMD 块
  for (int i = 1; i < block_num; ++i) {
    curr_idxs = simd_add(curr_idxs, idx_inc); // 更新当前块的真实基础下标
    simd_reg curr_vals = simd_load(M + i * simd_width);

#if defined(ENABLE_AVX512)
    // AVX-512: 使用 Mask 寄存器进行高效率的 Blend
    __mmask16 mask = _mm512_cmpgt_epi32_mask(curr_vals, max_vals);
    max_vals = _mm512_mask_blend_epi32(mask, max_vals, curr_vals);
    max_idxs = _mm512_mask_blend_epi32(mask, max_idxs, curr_idxs);
#elif defined(ENABLE_AVX2)
    // AVX2: 使用 _mm256_blendv_epi8，利用 _mm256_cmpgt_epi32 生成的 32-bit 掩码进行条件赋值
    __m256i mask = _mm256_cmpgt_epi32(curr_vals, max_vals);
    max_vals = _mm256_blendv_epi8(max_vals, curr_vals, mask);
    max_idxs = _mm256_blendv_epi8(max_idxs, curr_idxs, mask);
#else
    // SSE2: 没有原生按位 Blend 指令，使用标准的位运算 (AND/ANDNOT/OR) 组合实现条件替换
    __m128i mask = _mm_cmpgt_epi32(curr_vals, max_vals);
    max_vals = _mm_or_si128(_mm_and_si128(mask, curr_vals),
                            _mm_andnot_si128(mask, max_vals));
    max_idxs = _mm_or_si128(_mm_and_si128(mask, curr_idxs),
                            _mm_andnot_si128(mask, max_idxs));
#endif
  }

  // 收尾阶段：水平归约 (Horizontal Reduction)
  // 将包含局部最大值及其下标的 SIMD 寄存器存入内存，进行小规模的标量归约
  alignas(SIMD_BYTES) int final_vals[simd_width];
  alignas(SIMD_BYTES) int final_idxs[simd_width];
  simd_store(final_vals, max_vals);
  simd_store(final_idxs, max_idxs);

  int best_idx = final_idxs[0];
  int best_val = final_vals[0];

  // 在 simd_width (通常为 4, 8 或 16) 个元素中找出真正的全局最大值
  for (int i = 1; i < simd_width; ++i) {
    // 处理逻辑：发现更大的值，或者值相等但下标更靠前的情况（完美还原标量版本的逻辑行为）
    if (final_vals[i] > best_val || 
       (final_vals[i] == best_val && final_idxs[i] < best_idx)) {
      best_val = final_vals[i];
      best_idx = final_idxs[i];
    }
  }

  return best_idx;
}

#endif
