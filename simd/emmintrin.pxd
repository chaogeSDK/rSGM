"""SSE2指令集（GCC编译器）"""

cdef extern from "xmmintrin.h":
    # 数据类型
    ctypedef float __m128

cdef extern from "emmintrin.h":
    # 数据类型
    ctypedef long long __m128i
    ctypedef double __m128d


    # Integer：算术、比较、逻辑、移位、内存分配
    __m128i _mm_adds_epu16(__m128i a, __m128i b) nogil
    __m128i _mm_subs_epu16(__m128i a, __m128i b) nogil

    __m128i _mm_load_si128(__m128i* mem_addr) nogil
    __m128i _mm_loadu_si128 (__m128i* mem_addr) nogil
    __m128i _mm_set1_epi16(short a) nogil
    __m128i _mm_setzero_si128() nogil

    void _mm_storeu_si128(__m128i* mem_addr, __m128i a) nogil
    int _mm_extract_epi16(__m128i a, int imm8) nogil

    __m128i _mm_xor_si128(__m128i a, __m128i b) nogil

    # DP：
    __m128d _mm_loadu_pd(double* mem_addr) nogil

