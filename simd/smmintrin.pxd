"""SSE4.1/4.2指令集（GCC编译器）"""

from simd.emmintrin cimport __m128i


cdef extern from "smmintrin.h":
    ctypedef signed long __int64

    __m128i _mm_min_epu16(__m128i a, __m128i b) nogil
    __m128i _mm_minpos_epu16(__m128i a) nogil

    int _mm_extract_epi32(__m128i a, const int imm8) nogil
    __int64 _mm_extract_epi64(__m128i a, const int imm8) nogil