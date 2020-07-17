cdef extern from "mm_malloc.h":
    # 内存分配与释放
    void* _mm_malloc(size_t size, size_t align) nogil
    void _mm_free(void * mem_addr) nogil