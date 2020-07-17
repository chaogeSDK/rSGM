"""MMX指令集（GCC编译器）"""

cdef extern from "mmintrin.h":
    ctypedef int __m64
