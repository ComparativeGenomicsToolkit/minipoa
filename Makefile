# Plain-make build of libminipoa.a, for embedding in projects that do not drive cmake.
#
# CMakeLists.txt is still the upstream build and still works; this exists because cactus builds
# every submodule with `make` and passes its SIMD choice through the environment, the same way
# abPOA is built there.
#
# SIMD is selected from the environment, highest first: avx512bw, avx2, sse41, sse2, armv8.
# Nothing set means probe the compiler for -mavx2 and fall back to -msse4.1.
#
# The -m flag and the -DENABLE_* macro MUST agree: simdpriv.h has no default beyond a silent
# 128-bit #else, so a mismatch compiles cleanly and gives a different SIMD_BYTES/simd_width.

CC        ?= cc
CXX       ?= c++
AR        ?= ar

SRC_DIR    = src
INC_DIR    = include
LIB_DIR    = lib
MINIPOALIB = $(LIB_DIR)/libminipoa.a

ifneq ($(avx512bw),)
	SIMD_FLAGS = -mavx512f -mavx512bw
	SIMD_DEF   = -DENABLE_AVX512
else ifneq ($(avx2),)
	SIMD_FLAGS = -mavx2
	SIMD_DEF   = -DENABLE_AVX2
else ifneq ($(sse41),)
	SIMD_FLAGS = -msse4.1
	SIMD_DEF   = -DENABLE_SSE2
else ifneq ($(sse2),)
	SIMD_FLAGS = -msse2
	SIMD_DEF   = -DENABLE_SSE2
else ifneq ($(armv8),)
	# simde maps the x86 intrinsics onto NEON; see include/simde.
	SIMD_FLAGS = -march=armv8-a+simd
	SIMD_DEF   = -DENABLE_SSE2 -DUSE_SIMDE -DSIMDE_ENABLE_NATIVE_ALIASES
else ifneq ($(aarch64),)
	SIMD_FLAGS = -march=armv8-a+simd
	SIMD_DEF   = -DENABLE_SSE2 -DUSE_SIMDE -DSIMDE_ENABLE_NATIVE_ALIASES
else
	SIMD_PROBE := $(shell $(CXX) -mavx2 -E -x c++ /dev/null > /dev/null 2>&1 && echo avx2)
	ifeq ($(SIMD_PROBE),avx2)
		SIMD_FLAGS = -mavx2
		SIMD_DEF   = -DENABLE_AVX2
	else
		SIMD_FLAGS = -msse4.1
		SIMD_DEF   = -DENABLE_SSE2
	endif
endif

OPT      ?= -O3
COMMON    = $(OPT) -I$(INC_DIR) $(SIMD_FLAGS) $(SIMD_DEF)

# Our own -std= goes LAST so it wins over anything inherited through CFLAGS/CXXFLAGS.  The .c
# files are klib and need GNU extensions; strict -std=c11 does not build them.
ALL_CXXFLAGS = $(CXXFLAGS) $(COMMON) -fopenmp -std=c++11
ALL_CFLAGS   = $(CFLAGS) $(COMMON) -std=gnu11

# Exactly the CMake library source list, plus the C wrapper.  main.cpp is deliberately absent:
# it belongs to the minipoa binary, and linking it into a library would collide with the host's
# own main().
CXX_SRC = $(SRC_DIR)/align.cpp $(SRC_DIR)/graph.cpp $(SRC_DIR)/file_io.cpp \
          $(SRC_DIR)/lchain.cpp $(SRC_DIR)/mem_alloc_utils.cpp $(SRC_DIR)/minimizer.cpp \
          $(SRC_DIR)/sequence.cpp $(SRC_DIR)/minipoa_c.cpp
C_SRC   = $(SRC_DIR)/kalloc.c $(SRC_DIR)/kstring.c $(SRC_DIR)/utils.c

OBJS = $(CXX_SRC:.cpp=.o) $(C_SRC:.c=.o)

.PHONY: default all libminipoa clean
default: libminipoa
all: libminipoa
libminipoa: $(MINIPOALIB)

$(MINIPOALIB): $(OBJS)
	mkdir -p $(LIB_DIR)
	$(AR) -csr $@ $(OBJS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(ALL_CXXFLAGS) -c $< -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(ALL_CFLAGS) -c $< -o $@

# Required, and not only as a courtesy: cactus's generic `subclean.%` rule runs `make clean` in
# every submodule that has a makefile, so a Makefile without this target breaks `make clean`
# across that whole tree -- including the very first step of its Dockerfile and CI job.
clean:
	rm -f $(SRC_DIR)/*.o $(MINIPOALIB)
