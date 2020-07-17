"""AVX512指令集（GCC编译器）"""
from simd.emmintrin cimport __m128, __m128i, __m128d

cdef extern from "avx512fintrin.h":
    ctypedef float __m512
    ctypedef long long __m512i
    ctypedef double __m512d


cdef extern from "avx512erintrin.h":
    pass