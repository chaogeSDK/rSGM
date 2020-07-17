from simd.mm_malloc cimport *
from simd.emmintrin cimport *
from simd.pmmintrin cimport *
from simd.tmmintrin cimport *
from simd.smmintrin cimport *
from simd.popcntintrin cimport *

cdef extern from "immintrin.h":
    ####################################################################################################
    ##########################################---avxintrin.h---#########################################
    ####################################################################################################
    ctypedef float __m256
    ctypedef long long __m256i
    ctypedef double __m256d

    # Move Unaligned Packed Integer Values
    __m256i _mm256_loadu_si256(__m256i* mem_addr) nogil
    void _mm256_storeu_si256(__m256i* mem_addr, __m256i b) nogil

    # Move Aligned Packed Integer Values
    __m256i _mm256_load_si256(__m256i* mem_addr) nogil
    void _mm256_store_si256(__m256i* mem_addr, __m256i a) nogil

    # Extract integer element selected by index from 256-bit packed integer array
    int _mm256_extract_epi8 (__m256i a, const int index) nogil
    int _mm256_extract_epi16(__m256i a, const int index) nogil
    int _mm256_extract_epi32(__m256i a, const int index) nogil
    __int64 _mm256_extract_epi64(__m256i a, const int index) nogil

    # Extract packed floating-point values
    __m128 _mm256_extractf128_ps(__m256 a, const int index) nogil
    __m128d _mm256_extractf128_pd(__m256d a, const int index) nogil
    __m128i _mm256_extractf128_si256(__m256i a, const int index) nogil

    # Return 256-bit vector with all elements initialized to specified scalar
    __m256i _mm256_set1_epi8(char a) nogil
    __m256i _mm256_set1_epi16(short a) nogil
    __m256i _mm256_set1_epi32(int a) nogil
    __m256i _mm256_set1_epi64x(long long a) nogil

    # Return a vector with all elements set to zero
    __m256i _mm256_setzero_si256() nogil


    ####################################################################################################
    #######################################---avx2intrin.h---###########################################
    ####################################################################################################
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

    __m256i _mm256_alignr_epi32(__m256i a, __m256i b, const int num) nogil