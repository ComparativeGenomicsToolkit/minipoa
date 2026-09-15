#include "mem_alloc_utils.h"
// No <mm_malloc.h>: nothing here uses _mm_malloc/_mm_free (it is posix_memalign/free), and gcc
// ships that header only for x86 targets, so including it broke every aarch64 consumer of the
// public headers -- mem_alloc_utils.h is reachable from minipoa.h via align.h.
#include <cstdlib>
#include <cerrno>
#include <stdexcept>
#include <string>
#include <sys/mman.h> // 内存管理相关 (mlock, munlock, madvise)
#include <iostream>
/**
 * 32字节对齐，并用大页减少消除缺页异常
 *
 * 返回值: 成功返回分配的内存指针，失败返回nullptr
 */
void alloc_aligned(void** mem_ptr, size_t alignment, size_t size) {
  int ret = posix_memalign(mem_ptr, alignment, size);
  // 检查分配是否成功
  if (ret != 0) {
    // 错误处理: 将errno设置为返回值
    std::string errorMsg;
    switch (ret) {
      case EINVAL:
        errorMsg = "Invalid alignment parameter: alignment must be a power of two and a multiple of sizeof(void*)";
        break;
      case ENOMEM:
        errorMsg = "Insufficient memory for allocation request";
        break;
      default:
        errorMsg = "Unknown memory allocation error: error code " + std::to_string(ret);
    }
    throw std::runtime_error("POA_SIMD memory allocation failed: " + errorMsg);
  }
}

/**
 * 释放对齐分配的内存
 *
 * 参数ptr: 由aligned_alloc分配的内存指针
 */
void free_aligned(void* ptr) {
  if (ptr) {
    free(ptr);
  }
}
