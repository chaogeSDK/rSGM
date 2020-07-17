
from simd.avxintrin cimport __m256, __m256i, __m256d
from simd.emmintrin cimport __m128, __m128i, __m128d

cdef extern from "avx2intrin.h":
    ctypedef signed long __int64

    # Integer 256-bit vector logical operations.
    __m256i _mm256_and_si256(__m256i a, __m256i b) nogil
    __m256i _mm256_andnot_si256(__m256i a, __m256i b) nogil
    __m256i _mm256_or_si256(__m256i a, __m256i b) nogil
    __m256i _mm256_xor_si256(__m256i a, __m256i b) nogil

    # Integer 256-bit vector arithmetic operations.
    __m256i _mm256_adds_epi8(__m256i a, __m256i b) nogil
    __m256i _mm256_adds_epu8(__m256i a, __m256i b) nogil
    __m256i _mm256_adds_epi16(__m256i a, __m256i b) nogil
    __m256i _mm256_adds_epu16(__m256i a, __m256i b) nogil

    __m256i _mm256_subs_epi8(__m256i a, __m256i b) nogil
    __m256i _mm256_subs_epu8(__m256i a, __m256i b) nogil
    __m256i _mm256_subs_epi16(__m256i a, __m256i b) nogil
    __m256i _mm256_subs_epu16(__m256i a, __m256i b) nogil


    # Integer 128/256-bit vector pack/blend/shuffle/insert/extract operations
    __m256i _mm256_alignr_epi8(__m256i a, __m256i b, const int num) nogil

    # Integer 256-bit vector MIN/MAX operations.
    __m256i _mm256_max_epi8(__m256i a, __m256i b) nogil
    __m256i _mm256_max_epi16(__m256i a, __m256i b) nogil
    __m256i _mm256_max_epi32(__m256i a, __m256i b) nogil
    __m256i _mm256_max_epu8(__m256i a, __m256i b) nogil
    __m256i _mm256_max_epu16(__m256i a, __m256i b) nogil
    __m256i _mm256_max_epu32(__m256i a, __m256i b) nogil

    __m256i _mm256_min_epi8(__m256i a, __m256i b) nogil
    __m256i _mm256_min_epi16(__m256i a, __m256i b) nogil
    __m256i _mm256_min_epi32(__m256i a, __m256i b) nogil
    __m256i _mm256_min_epu8(__m256i a, __m256i b) nogil
    __m256i _mm256_min_epu16(__m256i a, __m256i b) nogil
    __m256i _mm256_min_epu32(__m256i a, __m256i b) nogil