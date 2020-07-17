cdef extern from "popcntintrin.h":
    int _mm_popcnt_u32 (unsigned int a) nogil
    long long _mm_popcnt_u64 (unsigned long long a) nogil