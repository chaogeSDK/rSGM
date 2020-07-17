"""SSSE3指令集（GCC编译器）"""
from simd.emmintrin cimport __m128i

cdef extern from "tmmintrin.h":
    # 数据类型
    __m128i _mm_alignr_epi8(__m128i a, __m128i b, int count) nogil
