####################################distutils: language=c++##################################################
from __future__ import print_function
import cv2
import cv2.ximgproc
import time
import numpy as np
cimport numpy as cnp
from multiprocessing import Process,Pool
cimport cython
cimport openmp
from libc.stdint cimport *
from libc.stdio cimport printf, perror
from libc.stdlib cimport malloc, free, calloc, abs
from libc.string cimport memset, memcpy
from cython.parallel import prange, parallel

# from simd.mm_malloc cimport *
# from simd.emmintrin cimport *
# from simd.pmmintrin cimport *
# from simd.tmmintrin cimport *
# from simd.smmintrin cimport *
# from simd.avx512intrin cimport *
# from simd.popcntintrin cimport *
from simd.immintrin cimport *



DEF USE_SSE = 1
DEF USE_AVX = 0
DEF USE_AVX512 = 1


# SGM参数设置
cdef class stereoParametersSGM:
    cdef public int numDisparities, P1, P2
    cdef public int line_length
    cdef public int lrThreshold
    cdef public double uniquenessRatio
    cdef public bint doLRC, doRLC, subpixelRefine, median_filter, weighted_median_filter, NLM_filter
    cdef public int speckleWindowSize, speckleRange
    cdef public int cheight
    cdef public int cwidth
    cdef public int descriptor_type

    def __init__(self, int numDisparities=64, int P1=10, int P2=80, int cheight = 5, int cwidth = 5):
        self.numDisparities = numDisparities  # 视差搜索范围
        self.P1 = P1  # 用于近邻元素视差差异为1时的惩罚值
        self.P2 = P2  # 用于近邻元素视差差异大于1时的惩罚值
        self.cheight = cheight  # CT变换的窗口高度
        self.cwidth = cwidth   # CT变换的窗口宽度
        self.descriptor_type = 0   # 特征描述子类型
        self.line_length = 8  # cross census变换尺寸长度
        self.lrThreshold = 2  # 左右一致性检验阈值
        self.uniquenessRatio = 0.85  # 匹配筛选的比率阈值
        self.subpixelRefine = True  # 是否做亚像素插值
        self.doLRC = True  # 是否做左右一致性校验
        self.doRLC = False  # 是否做右左一致性校验
        self.median_filter = False  # 是否做中值滤波
        self.weighted_median_filter = False  # 是否做加权中值滤波
        self.NLM_filter = False  # 是否做NLM去噪
        self.speckleWindowSize = 100 # 噪声连通域尺寸阈值
        self.speckleRange = 1  # 连通域灰度差异阈值

    cpdef setNumDisparities(self, int numDisparites):
        self.numDisparities = numDisparites

    cpdef int getnumDisparities(self):
        return self.numDisparities

    cpdef setP1(self, int P1):
        self.P1 = P1

    cpdef int getP1(self):
        return self.P1

    cpdef setP2(self, int P2):
        self.P2 = P2

    cpdef int getP2(self):
        return self.P2

    cpdef setDescriptorType(self, int descriptor_type):
        self.descriptor_type = descriptor_type

    cpdef int getDescriptorType(self):
        return self.descriptor_type

    cpdef setLineLength(self, int line_length):
        self.line_length = line_length

    cpdef int getLineLength(self):
        return self.line_length

    cpdef setLRThreshold(self, int lrThreshold):
        self.lrThreshold = lrThreshold

    cpdef int getLRThreshold(self):
        return self.lrThreshold

    cpdef setUniquenessRatio(self, double uniquenessRatio):
        self.uniquenessRatio = uniquenessRatio

    cpdef double getUniquenessRatio(self):
        return self.uniquenessRatio

    cpdef setLRCcheck(self, bint isdo):
        self.doLRC = isdo

    cpdef bint getLRCcheck(self):
        return self.doLRC

    cpdef setRLCcheck(self, bint isdo):
        self.doRLC = isdo

    cpdef bint getRLCcheck(self):
        return self.doRLC

    cpdef setMedianFilter(self, bint isdo):
        self.median_filter = isdo

    cpdef bint getMedianFilter(self):
        return self.median_filter

    cpdef setWeightedMedianFilter(self, bint isdo):
        self.weighted_median_filter = isdo

    cpdef bint getWeightedMedianFilter(self):
        return self.weighted_median_filter

    cpdef setNLMFilter(self, bint isdo):
        self.NLM_filter = isdo

    cpdef bint getNLMFilter(self):
        return self.NLM_filter

    cpdef setSubRefine(self, bint isdo):
        self.subpixelRefine = isdo

    cpdef bint getSubRefine(self):
        return self.subpixelRefine

    cpdef setCHeight(self, cheight):
        self.cheight = cheight

    cpdef int getCheight(self):
        return self.cheight

    cpdef setCWidth(self, cwidth):
        self.cwidth = cwidth

    cpdef int getCWidth(self):
        return self.cwidth

    cpdef setSpeckleWindowSize(self, int speckle_size):
        self.speckleWindowSize = speckle_size

    cpdef int getSpeckleWindowSize(self):
        return self.speckleWindowSize

    cpdef setSpeckleRange(self, int speckle_range):
        self.speckleRange = speckle_range

    cpdef int getSpeckleRange(self):
        return self.speckleRange



cdef class stereoSGM:
    cdef int m_width, m_height
    cdef uint16_t*  m_S
    cdef stereoParametersSGM m_param
    cdef int16_t* disparity_map_left
    cdef int16_t* disparity_map_right

    def __cinit__(self, int width, int height, stereoParametersSGM param):
        self.m_height = height
        self.m_width = width
        self.m_param = param
        self.disparity_map_left  = <int16_t*> calloc(self.m_height*self.m_width, sizeof(int16_t))
        self.disparity_map_right = <int16_t*> calloc(self.m_height*self.m_width, sizeof(int16_t))
        self.m_S = <uint16_t*> calloc(self.m_height*self.m_width*self.m_param.numDisparities, sizeof(uint16_t))

    def __dealloc__(self):
        free(self.disparity_map_left)
        free(self.disparity_map_right)
        free(self.m_S)

    cdef void process(self, unsigned short* dsi):
        # print('<4>代价聚合......')
        # tic = time.time()
        IF USE_AVX == 1:
            aggregate_costs_AVX(dsi, self.m_S, self.m_width, self.m_height, self.m_param.P1, self.m_param.P2, self.m_param.numDisparities)
        ELSE:
            aggregate_costs_SSE(dsi, self.m_S, self.m_width, self.m_height, self.m_param.P1, self.m_param.P2, self.m_param.numDisparities)

        # toc = time.time()
        # print('\t用时{:.3f}s'.format(toc-tic))
        #
        # print('<5>计算右视差图......')
        # tic = time.time()
        matchWTARight(self.m_S, self.disparity_map_right, self.m_width, self.m_height, self.m_param.numDisparities)
        # toc = time.time()
        # print('\t用时{:.3f}s'.format(toc-tic))
        #
        # print('<6>计算左视差图......')
        # tic = time.time()
        matchWTA(self.m_S, self.disparity_map_left, self.disparity_map_right,self.m_width, self.m_height,self.m_param.numDisparities, self.m_param.uniquenessRatio, self.m_param.lrThreshold,self.m_param.subpixelRefine, self.m_param.doLRC)
        # toc = time.time()
        # print('\t用时{:.3f}s'.format(toc-tic))

    cdef int16_t* get_disparity_map_left(self):
        return self.disparity_map_left

    cdef int16_t* get_disparity_map_right(self):
        return self.disparity_map_right



cdef class stereoRSGM:
    cdef public int width, height
    cdef public int numStrips, border
    cdef stereoParametersSGM param
    cdef displ, dispr

    def __init__(self, int width, int height, stereoParametersSGM param, int numStrips = 6, int border = 10):
        self.height = height
        self.width = width
        self.param = param
        self.numStrips = numStrips
        self.border = border
        self.displ = None
        self.dispr = None


    def compute(self, unsigned char[:,:] left, unsigned char[:,:] right):
        cdef uint16_t* dsi = calcCost_SSE(left, right, self.param)
        cdef int16_t* disparity_map_left  = <int16_t*> calloc(self.height*self.width, sizeof(int16_t))
        cdef int16_t* disparity_map_right = <int16_t*> calloc(self.height*self.width, sizeof(int16_t))

        cdef Py_ssize_t n
        cdef int start, end
        cdef int start_line, end_line
        cdef int upper_border, lower_border
        cdef int strip_height, strip_width = self.width
        cdef int16_t* strip_displ
        cdef int16_t* strip_dispr
        cdef stereoSGM sgm

        if self.numStrips <= 1:
            sgm = stereoSGM(self.width, self.height, self.param)
            sgm.process(dsi)
            strip_displ = sgm.get_disparity_map_left()
            strip_dispr = sgm.get_disparity_map_right()
            memcpy(disparity_map_left, strip_displ, sizeof(int16_t)*self.width*self.height)
            memcpy(disparity_map_right, strip_dispr, sizeof(int16_t)*self.width*self.height)
        else:
            for n in range(self.numStrips):
                start_line = n * (self.height // self.numStrips)
                end_line = (n + 1) * (self.height // self.numStrips) - 1
                if n == self.numStrips - 1:
                    end_line = self.height - 1
                upper_border = max(start_line - self.border, 0)
                lower_border = min(end_line + self.border, self.height - 1)
                strip_height = lower_border - upper_border + 1

                sgm = stereoSGM(strip_width, strip_height, self.param)
                sgm.process(dsi + upper_border * self.width * self.param.numDisparities)
                strip_displ = sgm.get_disparity_map_left()
                strip_dispr = sgm.get_disparity_map_right()

                if n == 0:
                    start = 0
                    end   = strip_height - 1 - self.border
                elif n == self.numStrips - 1:
                    start = self.border
                    end   = strip_height - 1
                else:
                    start = self.border
                    end   = strip_height - 1 - self.border
                memcpy(disparity_map_left + start_line * self.width, strip_displ + start * strip_width, sizeof(int16_t)*strip_width*(end-start+1))
                memcpy(disparity_map_right + start_line * self.width, strip_dispr + start * strip_width, sizeof(int16_t)*strip_width*(end-start+1))

        cdef cnp.int16_t[:,:] disparity_map_left_view = <cnp.int16_t[:self.height, :self.width]> disparity_map_left
        cdef cnp.int16_t[:,:] disparity_map_right_view = <cnp.int16_t[:self.height, :self.width]> disparity_map_right
        self.displ = np.asarray(disparity_map_left_view)
        self.dispr = np.asarray(disparity_map_right_view)

        # 中值滤波
        if self.param.median_filter == True:
            self.displ = cv2.medianBlur(self.displ, 3)

        # 加权中值滤波
        # if self.param.weighted_median_filter == True:
        #     self.displ = cv2.ximgproc.weightedMedianFilter(iml, self.displ, 3, sigma=20, weightType=cv2.ximgproc.WMF_EXP)

        # NLM 去噪
        if self.param.NLM_filter == True:
            self.displ.astype(np.uint16)
            self.displ = cv2.fastNlMeansDenoising(self.displ, h= [10], templateWindowSize=7, searchWindowSize=21, normType= cv2.NORM_L1)

        # 连通域噪声剔除
        if self.param.speckleWindowSize > 0:
            newVal = -16
            maxSpeckleSize = self.param.speckleWindowSize
            maxDiff = 16 * self.param.speckleRange
            self.displ, _ = cv2.filterSpeckles(self.displ, newVal, maxSpeckleSize, maxDiff)


    def get_disparity_map_left(self):

        return self.displ

    def get_disparity_map_right(self):

        return self.dispr

    def get_maxDisp(self):
        pass

    def set_stereoParameters(self):
        pass




# 计算汉明距离(16位)
cdef inline int hammdist(int a, int b) nogil:
    cdef signed short val
    cdef int dist = 0

    val = a ^ b

    # 计算1的个数
    while(val):
        dist += 1
        val &= val - 1

    return dist


# 位1计数查找表(用于16位)
def fill_popCount16LUT():
    cdef int popCountLUT[65535 + 1]
    cdef int i

    for i in range(65535 + 1):
        popCountLUT[i] = hammdist(i, 0)

    return popCountLUT


# 位1计数(64位)---查找表法
cpdef popCount64(long long x, popCountLUT):
    cdef unsigned char dist
    dist = popCountLUT[x & 0xFFFF] + popCountLUT[(x>>16) & 0xFFFF] + \
           popCountLUT[(x>>32) & 0xFFFF] + popCountLUT[x>>48]

    return dist



# 位1计数查找表(用于16位)
cdef int* fill_popCount16LUT_cFunc() nogil:
    cdef int i
    cdef int *popcountLUT = <int*> malloc((65535 + 1)*sizeof(int))

    for i in range(65535 + 1):
        popcountLUT[i] = hammdist(i, 0)

    return popcountLUT



# census变换
@cython.boundscheck(False)
@cython.wraparound(False)
def census_transform(unsigned char[:, :] image, int cheight, int cwidth):
    # 转换为单通道图
    if image.ndim >= 3:
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayImage = image
    cdef unsigned char[:,::1] grayImage_view = grayImage

    cdef int width, height
    cdef int x_offset, y_offset

    height = <int>grayImage_view.shape[0]
    width  = <int>grayImage_view.shape[1]
    x_offset = cheight // 2
    y_offset = cwidth // 2

    cdef cnp.ndarray census_image = np.zeros((height, width), dtype= np.uint8)
    cdef cnp.ndarray census_feature = np.zeros((height, width), dtype= np.uint64)
    cdef unsigned char[:, ::1] census_image_view = census_image
    cdef unsigned long long[:, ::1] census_feature_view = census_feature

    cdef Py_ssize_t x, y, i, j
    cdef unsigned long long census_value, bit
    cdef unsigned char center_pixel
    cdef long comparison

    for x in prange(x_offset, height - x_offset, nogil= True, schedule='static', chunksize=1):
        for y in range(y_offset, width - y_offset):
            census_value = 0
            center_pixel = grayImage_view[x,y]   #窗口中心像素值

            for i in range(cheight):
                for j in range(cwidth):
                    if i == x_offset and j == y_offset:
                        continue
                    else:
                        census_value = census_value << 1
                        comparison = grayImage_view[x - x_offset + i, y - y_offset + j] - center_pixel
                        if comparison < 0:
                            bit = 1
                        else:
                            bit = 0
                        census_value = census_value | bit

            census_feature_view[x, y] = census_value
            census_image_view[x, y] = <unsigned char>census_value


    return census_feature, census_image




# 计算代价立方体(左侧)
@cython.boundscheck(False)
@cython.wraparound(False)
def compute_cost_volume(unsigned char[:, :] left, unsigned char[:, :] right, param):
    assert left.ndim == 2 or right.ndim == 2, '仅支持灰度图！'
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], '左右视图必须有相同的形状！'
    assert param.numDisparities > 0, '视差搜索范围必须大于0！'

    cdef int height, width
    cdef int cheight, cwidth
    cdef int nDisp
    cdef int x_offset, y_offset
    cdef int* popcntLUT
    cdef int descripotor
    cdef int line_length

    height, width = left.shape[0:2]
    nDisp = param.numDisparities
    descriptor = param.descriptor_type
    line_length = param.line_length
    cheight = param.cheight
    cwidth = param.cwidth
    x_offset = cheight // 2
    y_offset = cwidth // 2
    popcntLUT = fill_popCount16LUT_cFunc()

    # 计算census特征
    cdef uint64_t* left_census_feature
    cdef uint64_t* right_census_feature
    print('<1>计算左census特征......')
    tic = time.time()
    if descriptor == 0:
        left_census_feature = census_9x7(left, cwidth, cheight)
    elif descriptor == 1:
        left_census_feature = symmetricCensus(left, cwidth, cheight)
    elif descriptor == 2:
        left_census_feature = crossSymmetricCensus(left, line_length)
    elif descriptor == 3:
        left_census_feature = crossCensus(left, line_length)
    # left_census_feature, _ = census_transform(left, param.csize)
    toc = time.time()
    print('\t用时{:.3f}s'.format(toc-tic))

    print('<2>计算右census特征......')
    tic = time.time()
    if descriptor == 0:
        right_census_feature = census_9x7(right, cwidth, cheight)
    elif descriptor == 1:
        right_census_feature = symmetricCensus(right, cwidth, cheight)
    elif descriptor == 2:
        right_census_feature = crossSymmetricCensus(right, line_length)
        x_offset = 0
        y_offset = 0
    elif descriptor == 3:
        right_census_feature = crossCensus(right, line_length)
        x_offset = 0
        y_offset = 0
    # right_census_feature, _ = census_transform(right, param.csize)
    toc = time.time()
    print('\t用时{:.3f}s'.format(toc - tic))


    # 计算代价立方体
    print('<3>计算代价立方体......')
    tic = time.time()

    # 初始化
    cdef unsigned short *cost_volume_c = <unsigned short*> calloc(height*width*nDisp, sizeof(unsigned short))
    cdef cnp.uint64_t[:, :] left_census_feature_view  = <cnp.uint64_t[:height, :width]> left_census_feature
    cdef cnp.uint64_t[:, :] right_census_feature_view = <cnp.uint64_t[:height, :width]> right_census_feature
    cdef uint64_t xor
    cdef unsigned short dist
    cdef Py_ssize_t x, y, d
    cdef cnp.uint16_t [:, :, :] cost_volume_view

    for x in prange(x_offset, height - x_offset, nogil=True, schedule='static', chunksize=1):
        for y in range(y_offset, width - y_offset):
            for d in range(nDisp):
                if y >= y_offset + d:
                    xor = left_census_feature_view[x, y] ^ right_census_feature_view[x, y - d]
                    dist = popcntLUT[xor & 0xFFFF] + popcntLUT[(xor>>16) & 0xFFFF] + popcntLUT[(xor>>32) & 0xFFFF] + popcntLUT[xor>>48]
                    cost_volume_c[x*width*nDisp + y*nDisp + d] = dist
                else:
                    xor = left_census_feature_view[x, y] ^ right_census_feature_view[x, y + nDisp -d]
                    dist = popcntLUT[xor & 0xFFFF] + popcntLUT[(xor>>16) & 0xFFFF] + popcntLUT[(xor>>32) & 0xFFFF] + popcntLUT[xor>>48]
                    cost_volume_c[x*width*nDisp + y*nDisp + d] = dist

    # 将C指针指向的数据转换为Numpy array
    cost_volume_view = <cnp.uint16_t[:height, :width, :nDisp]> cost_volume_c
    cost_volume = np.asarray(cost_volume_view)
    toc = time.time()
    print('\t用时{:.3f}s'.format(toc - tic))

    free(popcntLUT)

    return cost_volume



# AD cost
@cython.boundscheck(False)
@cython.wraparound(False)
def ADcost(unsigned char[:,:,:] left, unsigned char[:,:,:] right, int numDisparities):
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], '左右视图必须有相同的形状！'
    assert numDisparities > 0 and numDisparities % 16 == 0, '视差搜索范围必须大于0且为16的整数倍！'

    cdef int height = left.shape[0]
    cdef int width  = left.shape[1]
    cdef Py_ssize_t x, y, d
    cdef int16_t dist
    cdef uint16_t* cost_volume_c = <uint16_t*> calloc(height*width*numDisparities, sizeof(uint16_t))
    for x in range(height):
        for y in range(width):
            for d in range(numDisparities):
                if y - d >= 0:
                    dist = abs(left[x,y,0] - right[x,y-d,0]) + abs(left[x,y,1] - right[x,y-d,1]) + abs(left[x,y,2] - right[x,y-d,2])
                    cost_volume_c[x*width*numDisparities + y*numDisparities + d] = dist
                else:
                    cost_volume_c[x*width*numDisparities + y*numDisparities + d] = 0

    # 将C指针指向的数据转换为Numpy array
    cost_volume_view = <cnp.uint16_t[:height, :width, :numDisparities]> cost_volume_c
    cost_volume = np.asarray(cost_volume_view)

    return cost_volume


# census变换
@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint64_t* census_9x7(unsigned char[:, :] image, int cwidth, int cheight):
    cdef int width, height
    cdef int x_offset, y_offset

    height = image.shape[0]
    width  = image.shape[1]
    x_offset = cheight // 2
    y_offset = cwidth // 2

    cdef Py_ssize_t x, y, i, j
    cdef uint64_t census_value, bit
    cdef uint8_t center_pixel
    cdef int16_t comparison

    cdef uint64_t* census_feature = <uint64_t*> calloc(height*width, sizeof(uint64_t))
    for x in prange(x_offset, height - x_offset, nogil= True, schedule='static', chunksize=1):
        for y in range(y_offset, width - y_offset):
            census_value = 0
            center_pixel = image[x,y]   #窗口中心像素值

            for i in range(cheight):
                for j in range(cwidth):
                    if i == x_offset and j == y_offset:
                        continue
                    else:
                        census_value = census_value << 1
                        comparison = image[x - x_offset + i, y - y_offset + j] - center_pixel
                        if comparison < 0:
                            bit = 1
                        else:
                            bit = 0
                        census_value = census_value | bit

            census_feature[x*width + y] = census_value

    return census_feature


# symmetric census变换
@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint64_t* symmetricCensus(unsigned char[:, :] image, int cwidth, int cheight):
    cdef int height, width
    cdef int x_offset, y_offset

    height = image.shape[0]
    width  = image.shape[1]
    x_offset = cheight // 2
    y_offset = cwidth // 2

    cdef Py_ssize_t x, y, i, j
    cdef uint8_t center_pixel
    cdef uint64_t census_value, bit

    cdef uint64_t* census_feature = <uint64_t*> calloc(height*width, sizeof(uint64_t))
    for x in prange(x_offset, height - x_offset, nogil= True, schedule='static', chunksize=1):
        for y in range(y_offset, width - y_offset):
            census_value = 0
            for i in range(-x_offset, 0):
                for j in range(-y_offset, y_offset + 1):
                    census_value = census_value << 1
                    if image[x+i, y+j] <= image[x-i, y-j]:
                        bit = 0
                    else:
                        bit = 1

                    census_value = census_value | bit
            i = 0
            for j in range(-y_offset, 0):
                census_value = census_value << 1
                if image[x, y+j] <= image[x, y-j]:
                    bit = 0
                else:
                    bit = 1

                census_value = census_value | bit

            census_feature[x*width + y] = census_value

    return census_feature





# cross symmetric census变换
@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint64_t* crossSymmetricCensus(unsigned char[:, :] image, int line_length):
    cdef int height, width

    height = image.shape[0]
    width = image.shape[1]

    cdef Py_ssize_t x, y, k
    cdef uint64_t census_value, bit
    if line_length > 16:
        line_length = 16

    cdef uint64_t* census_feature = <uint64_t*> calloc(height*width, sizeof(uint64_t))
    for x in prange(0, height, nogil= True, schedule='static', chunksize=1):
        for y in range(0, width):
            census_value = 0
            for k in range(1, line_length + 1):
                # 左-右
                census_value = census_value << 1
                if x - k < 0 or x + k > height-1:
                    bit = 0
                elif image[x-k, y] <= image[x+k, y]:
                    bit = 0
                else:
                    bit = 1
                census_value = census_value | bit

                # 左上-右下
                census_value = census_value << 1
                if x - k < 0 or y - k < 0 or x + k > height-1 or y + k > width-1:
                    bit = 0
                elif image[x-k, y-k] <= image[x+k, y+k]:
                    bit = 0
                else:
                    bit = 1
                census_value = census_value | bit

                # 上-下
                census_value = census_value << 1
                if y - k < 0 or y + k > width-1:
                    bit = 0
                elif image[x, y-k] <= image[x, y+k]:
                    bit = 0
                else:
                    bit = 1
                census_value = census_value | bit

                # 右上-左下
                census_value = census_value << 1
                if x - k < 0 or y - k < 0 or x + k > height-1 or y + k > width-1:
                    bit = 0
                elif image[x+k, y-k] <= image[x-k, y+k]:
                    bit = 0
                else:
                    bit = 1
                census_value = census_value | bit

            census_feature[x*width + y] = census_value


    return census_feature



# cross census变换
@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint64_t* crossCensus(unsigned char[:, :] image, int line_length):
    cdef int height, width

    height = image.shape[0]
    width = image.shape[1]

    cdef Py_ssize_t x, y, k
    cdef uint64_t census_value, bit
    cdef uint8_t center_pixel
    if line_length > 8:
        line_length = 8

    cdef uint64_t* census_feature = <uint64_t*> calloc(height*width, sizeof(uint64_t))
    for x in prange(0, height, nogil= True, schedule='static', chunksize=1):
        for y in range(0, width):
            center_pixel = image[x, y]
            census_value = 0
            for k in range(1, line_length + 1):
                # 左-右
                census_value = census_value << 1
                if x - k < 0:
                    bit = 0
                elif image[x-k, y] < center_pixel:
                    bit = 1
                else:
                    bit = 0
                census_value = census_value | bit

                census_value = census_value << 1
                if x + k > height-1:
                    bit = 0
                elif image[x+k, y] < center_pixel:
                    bit = 1
                else:
                    bit= 0
                census_value = census_value | bit

                # 左上-右下
                census_value = census_value << 1
                if x - k < 0 or y - k < 0:
                    bit = 0
                elif image[x-k, y-k] < center_pixel:
                    bit = 1
                else:
                    bit = 0
                census_value = census_value | bit

                census_value = census_value << 1
                if x + k > height-1 or y + k > width-1:
                    bit = 0
                elif image[x+k, y+k] < center_pixel:
                    bit = 1
                else:
                    bit = 0
                census_value = census_value | bit

                # 上-下
                census_value = census_value << 1
                if  y - k < 0:
                    bit = 0
                elif image[x, y-k] < center_pixel:
                    bit = 1
                else:
                    bit = 0
                census_value = census_value | bit

                census_value = census_value << 1
                if y + k > width-1:
                    bit = 0
                elif image[x, y+k] < center_pixel:
                    bit = 1
                else:
                    bit = 0
                census_value = census_value | bit

                # 右上-左下
                census_value = census_value << 1
                if  x + k > height-1 or y - k < 0:
                    bit = 0
                elif image[x+k, y-k] < center_pixel:
                    bit = 1
                else:
                    bit = 0
                census_value = census_value | bit

                census_value = census_value << 1
                if x - k < 0 or y + k > width-1:
                    bit = 0
                elif image[x-k, y+k] < center_pixel:
                    bit = 1
                else:
                    bit = 0
                census_value = census_value | bit

            census_feature[x*width + y] = census_value


    return census_feature



# Hanmming-census代价
@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint16_t* calcCost_SSE(unsigned char[:, :] left, unsigned char[:, :] right, stereoParametersSGM param):
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], '左右视图必须有相同的形状！'
    assert param.numDisparities > 0 and param.numDisparities % 16 == 0, '视差搜索范围必须大于0且必须为16的整数倍！'

    cdef int height, width
    cdef int cheight, cwidth
    cdef int nDisp
    cdef int x_offset, y_offset
    cdef int line_length
    cdef int descripotor
    cdef int* popcntLUT

    height, width = left.shape[0:2]
    nDisp = param.numDisparities
    line_length = param.line_length
    descriptor = param.descriptor_type
    cheight = param.cheight
    cwidth = param.cwidth
    x_offset = cheight // 2
    y_offset = cwidth // 2

    # 计算census特征
    cdef uint64_t* left_census_feature
    cdef uint64_t* right_census_feature
    # print('<1>计算左census特征......')
    # tic = time.time()
    if descriptor == 0:
        left_census_feature = census_9x7(left, cwidth, cheight)
    elif descriptor == 1:
        left_census_feature = symmetricCensus(left, cwidth, cheight)
    elif descriptor == 2:
        left_census_feature = crossSymmetricCensus(left, line_length)
    elif descriptor == 3:
        left_census_feature = crossCensus(left, line_length)
    else:
        perror("不支持的特征描述子......")

    # toc = time.time()
    # print('\t用时{:.3f}s'.format(toc-tic))

    # print('<2>计算右census特征......')
    # tic = time.time()
    if descriptor == 0:
        right_census_feature = census_9x7(right, cwidth, cheight)
    elif descriptor == 1:
        right_census_feature = symmetricCensus(right, cwidth, cheight)
    elif descriptor == 2:
        right_census_feature = crossSymmetricCensus(right, line_length)
        x_offset = 0
        y_offset = 0
    elif descriptor == 3:
        right_census_feature = crossCensus(right, line_length)
        x_offset = 0
        y_offset = 0
    else:
        perror("不支持的特征描述子......")
    # toc = time.time()
    # print('\t用时{:.3f}s'.format(toc - tic))

    IF USE_AVX == 1:
        cdef int width2
        cdef int pad = 4 - (width - y_offset) % 4
        if pad == 4:
            width2 = width
        else:
            width2 = width + pad
    ELSE:
        cdef int width2
        cdef int pad = 2 - (width - y_offset) % 2
        if pad == 2:
            width2 = width
        else:
            width2 = width + pad

    cdef uint64_t* left_census_feature_pad =  <uint64_t*> calloc(height*width2, sizeof(uint64_t))
    cdef uint64_t* right_census_feature_pad = <uint64_t*> calloc(height*width2, sizeof(uint64_t))
    cdef Py_ssize_t off
    for off in range(height):
        memcpy(&left_census_feature_pad[off * width2], &left_census_feature[off * width], sizeof(uint64_t)*width)
        memcpy(&right_census_feature_pad[off * width2], &right_census_feature[off * width], sizeof(uint64_t)*width)


    # 计算代价立方体
    # print('<3>计算代价立方体......')
    # tic = time.time()
    # 初始化
    cdef uint16_t* cost_volume = <uint16_t*> calloc(height*width*nDisp, sizeof(uint16_t))
    cdef uint64_t censusl, censusr, xor
    cdef Py_ssize_t x, y, d

    IF USE_AVX == 1:
        cdef __m256i censusl4, censusr4, xor4
        for x in prange(x_offset, height - x_offset, nogil=True, schedule='static', chunksize=1):
            for y in range(y_offset, width - y_offset, 4):
                for d in range(nDisp):
                    if y >= y_offset + d:
                        censusl4 = _mm256_loadu_si256(<__m256i*> &left_census_feature_pad[x*width2 + y])
                        censusr4 = _mm256_loadu_si256(<__m256i*> &right_census_feature_pad[x*width2 + y - d])
                        xor4 = _mm256_xor_si256(censusl4, censusr4)
                        cost_volume[x*width*nDisp + y*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm256_extract_epi64(xor4, 0))
                        if y + 1 < width - y_offset:
                            cost_volume[x*width*nDisp + (y+1)*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm256_extract_epi64(xor4, 1))
                        if y + 2 < width - y_offset:
                            cost_volume[x*width*nDisp + (y+2)*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm256_extract_epi64(xor4, 2))
                        if y + 3 < width - y_offset:
                            cost_volume[x*width*nDisp + (y+3)*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm256_extract_epi64(xor4, 3))
                    elif y + 3 > y_offset + d:
                        censusl = left_census_feature_pad[x*width2 + y]
                        censusr = right_census_feature_pad[x*width2 + y + nDisp - d]
                        xor = censusl ^ censusr
                        cost_volume[x*width*nDisp + y*nDisp + d] = <uint16_t>_mm_popcnt_u64(xor)

                        if y + 1 >= y_offset + d:
                            censusl = left_census_feature_pad[x*width2 + y + 1]
                            censusr = right_census_feature_pad[x*width2 + y + 1 - d]
                            xor = censusl ^ censusr
                            cost_volume[x*width*nDisp + (y+1)*nDisp + d] = <uint16_t>_mm_popcnt_u64(xor)
                        else:
                            censusl = left_census_feature_pad[x*width2 + y + 1]
                            censusr = right_census_feature_pad[x*width2 + y + 1 + nDisp - d]
                            xor = censusl ^ censusr
                            cost_volume[x*width*nDisp + (y+1)*nDisp + d] = <uint16_t>_mm_popcnt_u64(xor)
                        if y + 2 >= y_offset + d:
                            censusl = left_census_feature_pad[x*width2 + y + 2]
                            censusr = right_census_feature_pad[x*width2 + y + 2 - d]
                            xor = censusl ^ censusr
                            cost_volume[x*width*nDisp + (y+2)*nDisp + d] = <uint16_t>_mm_popcnt_u64(xor)
                        else:
                            censusl = left_census_feature_pad[x*width2 + y + 2]
                            censusr = right_census_feature_pad[x*width2 + y + 2 + nDisp - d]
                            xor = censusl ^ censusr
                            cost_volume[x*width*nDisp + (y+2)*nDisp + d] = <uint16_t>_mm_popcnt_u64(xor)

                        censusl = left_census_feature_pad[x*width2 + y + 3]
                        censusr = right_census_feature_pad[x*width2 + y + 3 - d]
                        xor = censusl ^ censusr
                        cost_volume[x*width*nDisp + (y+3)*nDisp + d] = <uint16_t>_mm_popcnt_u64(xor)

                    else:
                        censusl4 = _mm256_loadu_si256(<__m256i*> &left_census_feature_pad[x*width2 + y])
                        censusr4 = _mm256_loadu_si256(<__m256i*> &right_census_feature_pad[x*width2 + y + nDisp - d])
                        xor4 = _mm256_xor_si256(censusl4, censusr4)
                        cost_volume[x*width*nDisp + y*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm256_extract_epi64(xor4, 0))
                        cost_volume[x*width*nDisp + (y+1)*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm256_extract_epi64(xor4, 1))
                        cost_volume[x*width*nDisp + (y+2)*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm256_extract_epi64(xor4, 2))
                        cost_volume[x*width*nDisp + (y+3)*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm256_extract_epi64(xor4, 3))
    ELSE:
        cdef __m128i censusl2, censusr2, xor2
        for x in prange(x_offset, height - x_offset, nogil=True, schedule='static', chunksize=1):
            for y in range(y_offset, width - y_offset, 2):
                for d in range(nDisp):
                    if y >= y_offset + d:
                        censusl2 = _mm_loadu_si128(<__m128i*> &left_census_feature_pad[x*width2 + y])
                        censusr2 = _mm_loadu_si128(<__m128i*> &right_census_feature_pad[x*width2 + y - d])
                        xor2 = _mm_xor_si128(censusl2, censusr2)
                        cost_volume[x*width*nDisp + y*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm_extract_epi64(xor2, 0))
                        if y + 1 < width - y_offset:
                            cost_volume[x*width*nDisp + (y+1)*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm_extract_epi64(xor2, 1))
                    elif y + 2 > y_offset + d:
                        censusl = left_census_feature_pad[x*width2 + y]
                        censusr = right_census_feature_pad[x*width2 + y + nDisp - d]
                        xor = censusl ^ censusr
                        cost_volume[x*width*nDisp + y*nDisp + d] = <uint16_t>_mm_popcnt_u64(xor)

                        censusl = left_census_feature_pad[x*width2 + y + 1]
                        censusr = right_census_feature_pad[x*width2 + y + 1 - d]
                        xor = censusl ^ censusr
                        cost_volume[x*width*nDisp + (y+1)*nDisp + d] = <uint16_t>_mm_popcnt_u64(xor)
                    else:
                        censusl2 = _mm_loadu_si128(<__m128i*> &left_census_feature_pad[x*width2 + y])
                        censusr2 = _mm_loadu_si128(<__m128i*> &right_census_feature_pad[x*width2 + y + nDisp - d])
                        xor2 = _mm_xor_si128(censusl2, censusr2)
                        cost_volume[x*width*nDisp + y*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm_extract_epi64(xor2, 0))
                        cost_volume[x*width*nDisp + (y+1)*nDisp + d] = <uint16_t>_mm_popcnt_u64(_mm_extract_epi64(xor2, 1))


    # toc = time.time()
    # print('\t用时{:.3f}s'.format(toc - tic))

    free(left_census_feature)
    free(right_census_feature)
    free(left_census_feature_pad)
    free(right_census_feature_pad)

    return cost_volume




cdef inline unsigned short* getAddress_S(unsigned short* S, int width, int numDisparities, int x, int y, int d):
    return S + x * width * numDisparities + y * numDisparities + d


cdef inline unsigned short saturate_cast_uint16(int v) nogil:
    if v <= 0:
        return 0
    elif v <= UINT16_MAX:
        return v
    else:
        return UINT16_MAX


# 交换指针
cdef inline void swapPointer(unsigned short** p1, unsigned short** p2) nogil:
    cdef unsigned short* temp
    temp = p1[0]
    p1[0] = p2[0]
    p2[0] = temp


# 自适应计算惩罚值
cdef int getP2(float alpha, int gamma, int minP2, unsigned char Ip, unsigned char Ipr):
    cdef int P2
    P2 = int(-alpha * abs(Ip - Ipr) + gamma)
    if P2 < minP2:
        P2 = minP2
    return P2


# 代价聚合
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef aggregate_costs(unsigned short[:,:,:] dsi, unsigned char[:,:] img, int Passes, signed int P1, signed int P2, int numDisparities):
    cdef int height = img.shape[0]
    cdef int width = img.shape[1]
    cdef int width2 = width + 2
    cdef int disp2 = numDisparities + 2
    cdef unsigned short MAX_SGM_COST = UINT16_MAX

    # 聚合代价立方体
    cdef unsigned short* S = <unsigned short*> malloc(width * height * numDisparities * sizeof(unsigned short))

    # 路径代价
    cdef unsigned short* L_r0      = <unsigned short*> malloc(disp2 * sizeof(unsigned short)) + 1  #用于临时存储
    cdef unsigned short* L_r0_last = <unsigned short*> malloc(disp2 * sizeof(unsigned short)) + 1
    cdef unsigned short* L_r1      = <unsigned short*> malloc(width2 * disp2 * sizeof(unsigned short)) + disp2 + 1  #用于临时存储
    cdef unsigned short* L_r1_last = <unsigned short*> malloc(width2 * disp2 * sizeof(unsigned short)) + disp2 + 1
    cdef unsigned short* L_r2      = <unsigned short*> malloc(width * disp2 * sizeof(unsigned short)) + 1  #用于临时存储
    cdef unsigned short* L_r2_last = <unsigned short*> malloc(width * disp2 * sizeof(unsigned short)) + 1
    cdef unsigned short* L_r3      = <unsigned short*> malloc(width2 * disp2 * sizeof(unsigned short)) + disp2 + 1  #用于临时存储
    cdef unsigned short* L_r3_last = <unsigned short*> malloc(width2 * disp2 * sizeof(unsigned short)) + disp2 + 1

    # 最低代价
    cdef unsigned short  minL_r0_array[2]
    cdef unsigned short* minL_r0      = &minL_r0_array[0]
    cdef unsigned short* minL_r0_last = &minL_r0_array[1]
    cdef unsigned short* minL_r1      = <unsigned short*> malloc(width2 * sizeof(unsigned short)) + 1
    cdef unsigned short* minL_r1_last = <unsigned short*> malloc(width2 * sizeof(unsigned short)) + 1
    cdef unsigned short* minL_r2      = <unsigned short*> malloc(width * sizeof(unsigned short))
    cdef unsigned short* minL_r2_last = <unsigned short*> malloc(width * sizeof(unsigned short))
    cdef unsigned short* minL_r3      = <unsigned short*> malloc(width2 * sizeof(unsigned short)) + 1
    cdef unsigned short* minL_r3_last = <unsigned short*> malloc(width2 * sizeof(unsigned short)) + 1

    # 图像行指针
    # cdef unsigned char* img_line = NULL
    # cdef unsigned char* img_line_last = NULL


    #边界赋值
    memset(&L_r1[-disp2], MAX_SGM_COST, sizeof(unsigned short)*disp2)
    memset(&L_r1_last[-disp2], MAX_SGM_COST, sizeof(unsigned short)*disp2)
    memset(&L_r3[-disp2], MAX_SGM_COST, sizeof(unsigned short)*disp2)
    memset(&L_r3_last[-disp2], MAX_SGM_COST, sizeof(unsigned short)*disp2)
    L_r1[-disp2 - 1] = L_r1_last[-disp2 - 1] = MAX_SGM_COST
    L_r3[-disp2 - 1] = L_r3_last[-disp2 - 1] = MAX_SGM_COST

    memset(&L_r1[width * disp2 - 1], MAX_SGM_COST, sizeof(unsigned short)*disp2)
    memset(&L_r1_last[width * disp2 - 1], MAX_SGM_COST, sizeof(unsigned short)*disp2)
    memset(&L_r3[width * disp2 - 1], MAX_SGM_COST, sizeof(unsigned short)*disp2)
    memset(&L_r3_last[width * disp2 - 1], MAX_SGM_COST, sizeof(unsigned short)*disp2)

    L_r0[-1] = L_r1[-1] = L_r2[-1] = L_r3[-1] = MAX_SGM_COST
    L_r0[numDisparities] = L_r1[numDisparities] = L_r2[numDisparities] = L_r3[numDisparities] = MAX_SGM_COST
    L_r0_last[-1] = L_r1_last[-1] = L_r2_last[-1] = L_r3_last[-1] = MAX_SGM_COST
    L_r0_last[numDisparities] = L_r1_last[numDisparities] = L_r2_last[numDisparities] = L_r3_last[numDisparities] = MAX_SGM_COST

    minL_r1[-1] =  minL_r1[width] = 0
    minL_r1_last[-1] = minL_r1_last[width] = 0
    minL_r3[-1] =  minL_r3[width] = 0
    minL_r3_last[-1] = minL_r3_last[width] = 0


    # 代价聚合,分成两个阶段进行：左上->右下、右下->左上
    cdef Py_ssize_t pass_t
    cdef Py_ssize_t i, j, d
    cdef int i1, i2, di
    cdef int j1, j2, dj
    cdef int adaP2
    cdef unsigned short cost, minCost, newCost, newCost_all
    cdef int mincostL, costp11, costp12, costp2, delta_mincost

    for pass_t in range(Passes):
        if pass_t == 0:
            i1, i2, di = 0, height, 1
            j1, j2, dj = 0, width,  1
        else:
            i1, i2, di = height-1, -1, -1
            j1, j2, dj = width-1, -1, -1

        # 第一行第一个像素
        minCost = MAX_SGM_COST
        if pass_t == 0:
            for d in range(numDisparities):
                cost = dsi[i1, j1, d]

                L_r0_last[d] = cost
                L_r1_last[j1*disp2 + d] = cost
                L_r2_last[j1*disp2 + d] = cost
                L_r3_last[j1*disp2 + d] = cost

                if cost < minCost:
                    minCost = cost
                S[i1 * width * numDisparities + j1 * numDisparities + d] = cost
        else:
            for d in range(numDisparities):
                cost = dsi[i1, j1, d]

                L_r0_last[d] = cost
                L_r1_last[j1*disp2 + d] = cost
                L_r2_last[j1*disp2 + d] = cost
                L_r3_last[j1*disp2 + d] = cost

                if cost < minCost:
                    minCost = cost
                S[i1 * width * numDisparities + j1 * numDisparities + d] += cost

        # 存储最低代价
        minL_r0_last[0] = minCost
        minL_r1_last[0] = minCost
        minL_r2_last[0] = minCost
        minL_r3_last[0] = minCost

        # 第一行剩余像素
        for j in range(j1+dj, j2, dj):
            minCost    = MAX_SGM_COST
            minL_r0[0] = MAX_SGM_COST

            for d in range(numDisparities):
                cost = dsi[i1, j, d]

                L_r1_last[j*disp2 + d] = cost
                L_r2_last[j*disp2 + d] = cost
                L_r3_last[j*disp2 + d] = cost

                if cost < minCost:
                    minCost = cost

                # 计算L_r0上的最低代价
                mincostL = L_r0_last[d]
                costp11  = L_r0_last[d - 1] + P1
                costp12  = L_r0_last[d + 1] + P1
                #adaP2 = getP2(0.5, 70, img[i1, j], img[i1, j-1], 25)
                costp2   = minL_r0_last[0] + P2
                if mincostL > costp11:
                    mincostL = costp11
                if mincostL > costp12:
                    mincostL = costp12
                if mincostL > costp2:
                    mincostL = costp2
                delta_mincost = mincostL - minL_r0_last[0]

                newCost = saturate_cast_uint16(cost + delta_mincost)
                L_r0[d] = newCost

                if minL_r0[0] > newCost:
                    minL_r0[0] = newCost

                if pass_t == 0:
                    S[i1*width*numDisparities + j*numDisparities + d] = newCost
                else:
                    S[i1*width*numDisparities + j*numDisparities + d] += newCost

            minL_r1_last[j] = minCost
            minL_r2_last[j] = minCost
            minL_r3_last[j] = minCost

            # 交换L0的缓存
            swapPointer(&L_r0, &L_r0_last)
            swapPointer(&minL_r0, &minL_r0_last)

            # 边界赋值
            L_r1[j*disp2 - 1] = L_r2[j*disp2 - 1] = L_r3[j*disp2 - 1] = MAX_SGM_COST
            L_r1[j*disp2 + numDisparities] = L_r2[j*disp2 + numDisparities] = L_r3[j*disp2 + numDisparities] = MAX_SGM_COST

            L_r1_last[j*disp2 - 1] = L_r2_last[j*disp2 - 1] = L_r3_last[j*disp2 - 1] = MAX_SGM_COST
            L_r1_last[j*disp2 + numDisparities] = L_r2_last[j*disp2 + numDisparities] = L_r3_last[j*disp2 + numDisparities] = MAX_SGM_COST


        # 剩余行聚合
        for i in range(i1+di, i2, di):
            memset(L_r0_last, 0, numDisparities*sizeof(unsigned short))
            minL_r0_last[0] = 0

            for j in range(j1, j2, dj):
                minL_r0[0] = MAX_SGM_COST
                minL_r1[j] = MAX_SGM_COST
                minL_r2[j] = MAX_SGM_COST
                minL_r3[j] = MAX_SGM_COST

                for d in range(numDisparities):
                    newCost = 0
                    newCost_all = 0
                    cost = dsi[i, j, d]

                    # L_r0方向进行代价聚合
                    mincostL = L_r0_last[d]
                    costp11  = L_r0_last[d-1] + P1
                    costp12  = L_r0_last[d+1] + P1
                    #adaP2 = getP2(0.5, 70, img[i, j], img[i, j-1], 25)
                    costp2   = minL_r0_last[0] + P2
                    if mincostL > costp11:
                        mincostL = costp11
                    if mincostL > costp12:
                        mincostL = costp12
                    if mincostL > costp2:
                        mincostL = costp2
                    delta_mincost = mincostL - minL_r0_last[0]
                    newCost = saturate_cast_uint16(cost + delta_mincost)
                    newCost_all += newCost

                    # 存储L_r0路径当前视差下的聚合代价
                    L_r0[d] = newCost
                    if minL_r0[0] > newCost:
                        minL_r0[0] = newCost

                    # L_r1方向进行代价聚合
                    mincostL = L_r1_last[(j-dj)*disp2 + d]
                    costp11 = L_r1_last[(j-dj)*disp2 + d - 1] + P1
                    costp12 = L_r1_last[(j-dj)*disp2 + d + 1] + P1
                    #adaP2 = getP2(0.5, 70, img[i, j], img[i-1, j-1], 25)
                    costp2 = minL_r1_last[j-dj] + P2
                    if mincostL > costp11:
                        mincostL = costp11
                    if mincostL > costp12:
                        mincostL = costp12
                    if mincostL > costp2:
                        mincostL = costp2
                    delta_mincost = mincostL - minL_r1_last[j-dj]
                    newCost = saturate_cast_uint16(cost + delta_mincost)
                    newCost_all += newCost

                    # 存储L_r1路径当前视差下的聚合代价
                    L_r1[j*disp2 + d] = newCost
                    if minL_r1[j] > newCost:
                        minL_r1[j] = newCost

                    # L_r2方向进行代价聚合
                    mincostL = L_r2_last[j*disp2 + d]
                    costp11 = L_r1_last[j*disp2 + d - 1] + P1
                    costp12 = L_r1_last[j*disp2 + d + 1] + P1
                    #adaP2 = getP2(0.5, 70, img[i, j], img[i-1, j], 25)
                    costp2 = minL_r1_last[j] + P2
                    if mincostL > costp11:
                        mincostL = costp11
                    if mincostL > costp12:
                        mincostL = costp12
                    if mincostL > costp2:
                        mincostL = costp2
                    delta_mincost = mincostL - minL_r2_last[j]
                    newCost = saturate_cast_uint16(cost + delta_mincost)
                    newCost_all += newCost

                    # 存储L_r2路径当前视差下的聚合代价
                    L_r2[j*disp2 + d] = newCost
                    if minL_r2[j] > newCost:
                        minL_r2[j] = newCost

                    # L_r3方向进行代价聚合
                    mincostL = L_r3_last[(j+dj)*disp2 + d]
                    costp11 = L_r3_last[(j+dj)*disp2 + d - 1] + P1
                    costp12 = L_r3_last[(j+dj)*disp2 + d + 1] + P1
                    #adaP2 = getP2(0.5, 70, img[i, j], img[i-1, j+1], 25)
                    costp2 = minL_r3_last[j+dj] + P2
                    if mincostL > costp11:
                        mincostL = costp11
                    if mincostL > costp12:
                        mincostL = costp12
                    if mincostL > costp2:
                        mincostL = costp2
                    delta_mincost = mincostL - minL_r3_last[j+dj]
                    newCost = saturate_cast_uint16(cost + delta_mincost)
                    newCost_all += newCost

                    # 存储L_r3路径当前视差下的聚合代价
                    L_r3[j*disp2 + d] = newCost
                    if minL_r3[j] > newCost:
                        minL_r3[j] = newCost

                    if pass_t == 0:
                        S[i*width*numDisparities + j*numDisparities + d] = newCost_all
                    else:
                        S[i*width*numDisparities + j*numDisparities + d] += newCost_all
                swapPointer(&L_r0, &L_r0_last)
                swapPointer(&minL_r0, &minL_r0_last)

            swapPointer(&L_r1, &L_r1_last)
            swapPointer(&L_r2, &L_r2_last)
            swapPointer(&L_r3, &L_r3_last)
            swapPointer(&minL_r1, &minL_r1_last)
            swapPointer(&minL_r2, &minL_r2_last)
            swapPointer(&minL_r3, &minL_r3_last)


    free(L_r0 - 1)
    free(L_r0_last - 1)
    free(L_r1 - disp2 - 1)
    free(L_r1_last - disp2 - 1)
    free(L_r2 - 1)
    free(L_r2_last - 1)
    free(L_r3 - disp2 - 1)
    free(L_r3_last - disp2 - 1)
    free(minL_r1 - 1)
    free(minL_r1_last - 1)
    free(minL_r2)
    free(minL_r2_last)
    free(minL_r3 - 1)
    free(minL_r3_last - 1)


    cdef cnp.uint16_t [:, :, :] aggcost_volume_view
    aggcost_volume_view = <cnp.uint16_t[:height, :width, :numDisparities]> S
    aggregation_volume = np.asarray(aggcost_volume_view)

    return aggregation_volume


cdef inline int16_t subpixel_interpolate(short disp, unsigned short c0, unsigned short c1, unsigned short c2) nogil:
    cdef int16_t subDisp
    cdef int16_t denom2, a, b

    a = c0 - c2
    b = c0 - 2*c1 + c2
    denom2 = max(b,1)
    subDisp = disp * 16 + (a * 16 + denom2) / (denom2 * 2)

    return subDisp



# 视差计算与ratio test、亚像素插值（左视差图）
cdef void matchWTA(uint16_t* cost_volume, int16_t* disparity_map_left, int16_t* disparity_map_right, int width, int height, int numDisparities, double uniquenessRatio, int lrThreshold, bint doSubRefine, bint doLRCcheck) nogil:

    cdef unsigned short cost, minCost, secMinCost
    cdef short subpixel_disp, bestDisp, bestDispr, secBestDisp = 0
    cdef Py_ssize_t i, j, d
    cdef unsigned short c0, c1, c2
    cdef int end
    cdef int diff
    cdef int factor = <int>(1024*uniquenessRatio)

    # if doLRCcheck:
    #     disparity_map_right = cmatchWTARight(cost_volume, width, height, numDisparities)

    for i in prange(height, nogil= True, schedule='static', chunksize=1):
        for j in range(width):
            minCost = 65535
            secMinCost = 65535
            bestDisp = 0

            if j < numDisparities - 1:
                # disparity_map[i*width + j] = -16
                # continue
                for d in range(1, j):
                    cost = cost_volume[i*width*numDisparities + j*numDisparities + d]
                    if cost < secMinCost:
                        if cost < minCost:
                            secMinCost = minCost
                            minCost = cost
                            secBestDisp = bestDisp
                            bestDisp = d
                        else:
                            secMinCost = cost
                            secBestDisp = d
            else:
                for d in range(numDisparities):
                    cost = cost_volume[i*width*numDisparities + j*numDisparities + d]
                    if cost < secMinCost:
                        if cost < minCost:
                            secMinCost = minCost
                            minCost = cost
                            secBestDisp = bestDisp
                            bestDisp = d
                        else:
                            secMinCost = cost
                            secBestDisp = d

            # 左右一致性检验
            if doLRCcheck:
                bestDispr = disparity_map_right[i*width + j - bestDisp]
                diff = abs(bestDisp - bestDispr)
                if diff > lrThreshold:
                    bestDisp = -16
                    disparity_map_left[i*width + j] = bestDisp
                    continue

            # 比率测试
            if 1024*minCost < factor*secMinCost or abs(bestDisp - secBestDisp) < 2:
                if 0 < bestDisp < numDisparities - 1:
                    # 亚像素插值
                    if doSubRefine:
                        c0 = cost_volume[i*width*numDisparities + j*numDisparities + bestDisp - 1]
                        c1 = minCost
                        c2 = cost_volume[i*width*numDisparities + j*numDisparities + bestDisp + 1]
                        subpixel_disp = subpixel_interpolate(bestDisp, c0, c1, c2)
                        disparity_map_left[i*width + j] = subpixel_disp
                    else:
                        disparity_map_left[i*width + j] = 16 * bestDisp
                else:
                    disparity_map_left[i*width + j] = 16 * bestDisp
            else:
                disparity_map_left[i*width + j] = -16




# 计算右视差图
cdef void matchWTARight(uint16_t* cost_volume, int16_t* disparity_map_right, int width, int height, int numDisparities) nogil:
    #对角线搜索
    cdef Py_ssize_t i, j, d
    cdef unsigned short cost, minCost
    cdef int16_t bestDisp = 0

    for i in prange(height, nogil= True, schedule='static', chunksize=1):
        for j in range(width):
            minCost = UINT16_MAX

            if j + numDisparities > width - 1:
                for d in range(width - j - 1):
                    cost = cost_volume[i*width*numDisparities + (j+d)*numDisparities + d]
                    if cost < minCost:
                        minCost = cost
                        bestDisp = d
                disparity_map_right[i*width + j] = bestDisp
            else:
                for d in range(numDisparities):
                    cost = cost_volume[i*width*numDisparities + (j+d)*numDisparities + d]
                    if cost < minCost:
                        minCost = cost
                        bestDisp = d
                disparity_map_right[i*width + j] = bestDisp



# 计算右视差图
def matchWTARightPy(unsigned short[:,:,:] cost_volume, int width, int height, int numDisparities):
    cdef int16_t* disparity_map_c = <int16_t*> malloc(width*height*sizeof(int16_t))

    #对角线搜索
    cdef Py_ssize_t i, j, d
    cdef unsigned short cost, minCost
    cdef int16_t bestDisp = 0

    for i in range(height):
        for j in range(width):
            minCost = UINT16_MAX

            if j + numDisparities > width - 1:
                for d in range(width - j - 1):
                    cost = cost_volume[i, j+d, d]
                    if cost < minCost:
                        minCost = cost
                        bestDisp = d
                disparity_map_c[i*width + j] = bestDisp
            else:
                for d in range(numDisparities):
                    cost = cost_volume[i, j+d, d]
                    if cost < minCost:
                        minCost = cost
                        bestDisp = d
                disparity_map_c[i*width + j] = bestDisp

    cdef cnp.int16_t[:,:] disparity_map_view
    disparity_map_view = <cnp.int16_t[:height, :width]> disparity_map_c
    disparity_map = np.asarray(disparity_map_view)

    return disparity_map


# 左右一致性检验
def LRCcheck(int16_t[:,:] disparity_map_left, int16_t[:,:] disparity_map_right, int lrThreshold):
    cdef int width, height
    cdef Py_ssize_t i, j
    cdef int diff

    height = disparity_map_left.shape[0]
    width  = disparity_map_left.shape[1]

    for i in range(height):
        for j in range(width):
            dl = disparity_map_left[i,j]

            if dl >= 0  and dl <= j:
                dr = disparity_map_right[i, <Py_ssize_t>(j - dl)]
                diff = abs(dl - dr)
                if diff > lrThreshold:
                    disparity_map_left[i,j] = -1


    return np.array(disparity_map_left)


# 右左一致性检验
def RLCcheck(int16_t[:,:] disparity_map_left, int16_t[:,:] disparity_map_right, int lrThreshold):
    cdef int width, height
    cdef Py_ssize_t i, j
    cdef int diff

    height = disparity_map_left.shape[0]
    width  = disparity_map_left.shape[1]

    for i in range(height):
        for j in range(width):
            dr = disparity_map_right[i, j]

            if dr >= 0 and j + dr <= width:
                dl = disparity_map_left[i, <Py_ssize_t>(j + dr)]
                diff = abs(dl - dr)
                if diff > lrThreshold:
                    disparity_map_right[i, j] = -1
            else:
                disparity_map_right[i,j] = -1

    return disparity_map_right



cdef void aggregate_costs_SSE(uint16_t* dsi, uint16_t* S, int width, int height, int P1, int P2, int numDisparities):
    # 支持sse2
    cdef int width2 = width + 2
    cdef int disp2  = numDisparities + (128 / 16)
    cdef uint16_t MAX_SGM_COST = UINT16_MAX
    cdef int Passes = 2

    # 聚合代价立方体
    # cdef uint16_t* S = <uint16_t*>malloc(width * height * numDisparities * sizeof(uint16_t))

    # 路径代价
    cdef uint16_t* L_r0      = <uint16_t*>_mm_malloc(disp2 * sizeof(uint16_t), 16) + 1
    cdef uint16_t* L_r0_last = <uint16_t*>_mm_malloc(disp2 * sizeof(uint16_t), 16) + 1
    cdef uint16_t* L_r1      = <uint16_t*>_mm_malloc(width2 * disp2 * sizeof(uint16_t), 16) + disp2 + 1
    cdef uint16_t* L_r1_last = <uint16_t*>_mm_malloc(width2 * disp2 * sizeof(uint16_t), 16) + disp2 + 1
    cdef uint16_t* L_r2      = <uint16_t*>_mm_malloc(width * disp2 * sizeof(uint16_t), 16) + 1
    cdef uint16_t* L_r2_last = <uint16_t*>_mm_malloc(width * disp2 * sizeof(uint16_t), 16) + 1
    cdef uint16_t* L_r3      = <uint16_t*>_mm_malloc(width2 * disp2 * sizeof(uint16_t), 16) + disp2 + 1
    cdef uint16_t* L_r3_last = <uint16_t*>_mm_malloc(width2 * disp2 * sizeof(uint16_t), 16) + disp2 + 1

    # 最低代价
    cdef uint16_t minL_r0_array[2]
    cdef uint16_t* minL_r0      = &minL_r0_array[0]
    cdef uint16_t* minL_r0_last = &minL_r0_array[1]
    cdef uint16_t* minL_r1      = <uint16_t*>_mm_malloc(width2*sizeof(uint16_t), 16) + 1
    cdef uint16_t* minL_r1_last = <uint16_t*>_mm_malloc(width2*sizeof(uint16_t), 16) + 1
    cdef uint16_t* minL_r2      = <uint16_t*>_mm_malloc(width*sizeof(uint16_t), 16)
    cdef uint16_t* minL_r2_last = <uint16_t*>_mm_malloc(width*sizeof(uint16_t), 16)
    cdef uint16_t* minL_r3      = <uint16_t*>_mm_malloc(width2*sizeof(uint16_t), 16) + 1
    cdef uint16_t* minL_r3_last = <uint16_t*>_mm_malloc(width2*sizeof(uint16_t), 16) + 1

    # 边界赋值
    memset(&L_r1[-disp2], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r1_last[-disp2], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r3[-disp2], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r3_last[-disp2], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    L_r1[-disp2-1] = L_r1_last[-disp2-1] = MAX_SGM_COST
    L_r3[-disp2-1] = L_r3_last[-disp2-1] = MAX_SGM_COST

    memset(&L_r1[width * disp2 - 1], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r1_last[width * disp2 - 1], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r3[width * disp2 - 1], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r3_last[width * disp2 - 1], MAX_SGM_COST, sizeof(uint16_t)*disp2)


    L_r0[-1] = L_r2[-1]= MAX_SGM_COST
    L_r0_last[-1] = L_r2_last[-1] = MAX_SGM_COST
    memset(&L_r0[numDisparities], MAX_SGM_COST, sizeof(uint16_t) * (8-1))
    memset(&L_r2[numDisparities], MAX_SGM_COST, sizeof(uint16_t) * (8-1))
    memset(&L_r0_last[numDisparities], MAX_SGM_COST, sizeof(uint16_t) * (8-1))
    memset(&L_r2_last[numDisparities], MAX_SGM_COST, sizeof(uint16_t) * (8-1))

    minL_r1[-1] = minL_r1_last[-1] = 0
    minL_r1[width] = minL_r1_last[width] = 0
    minL_r3[-1] = minL_r3_last[-1] = 0
    minL_r3[width] = minL_r3_last[width] = 0

    cdef Py_ssize_t pass_t
    cdef Py_ssize_t i, j, d
    cdef Py_ssize_t i1, i2, di
    cdef Py_ssize_t j1, j2, dj
    cdef uint16_t cost, minCost, newCost, newCost_all
    cdef int mincostL, costp11, costp12, costp2, delta_mincost

    cdef int baseindexr0, baseindexr1, baseindexr2, baseindexr3
    cdef __m128i minLr_08, minLr_18, minLr_28, minLr_38
    cdef __m128i minLr_last_08, minLr_last_18, minLr_last_28, minLr_last_38
    cdef __m128i costpm8_r0, costpm8_r1, costpm8_r2, costpm8_r3
    cdef __m128i costpp8_r0, costpp8_r1, costpp8_r2, costpp8_r3
    cdef __m128i costp28_r0, costp28_r1, costp28_r2, costp28_r3
    #cdef __m128i mincostp28_r0, mincostp28_r1, mincostp28_r2, mincostp28_r3
    cdef __m128i lower8_r0, lower8_r1, lower8_r2, lower8_r3
    cdef __m128i upper8_r0, upper8_r1, upper8_r2, upper8_r3
    cdef __m128i cost8, mincost8, newCost8, newCost8_all
    cdef __m128i P1_8 = _mm_set1_epi16(<int16_t> P1)
    cdef __m128i P2_8 = _mm_set1_epi16(<int16_t> P2)

    for pass_t in range(Passes):
        if pass_t == 0:
            i1, i2, di = 0, height, 1
            j1, j2, dj = 0, width,  1
        else:
            i1, i2, di = height-1, -1, -1
            j1, j2, dj = width-1, -1, -1

        # 第一行第一个像素
        minCost = MAX_SGM_COST
        if pass_t == 0:
            for d in range(numDisparities):
                cost = dsi[i1*width*numDisparities + j1*numDisparities + d]

                L_r0_last[d] = cost
                L_r1_last[j1*disp2 + d] = cost
                L_r2_last[j1*disp2 + d] = cost
                L_r3_last[j1*disp2 + d] = cost

                if cost < minCost:
                    minCost = cost
                S[i1 * width * numDisparities + j1 * numDisparities + d] = cost
        else:
            for d in range(numDisparities):
                cost = dsi[i1*width*numDisparities + j1*numDisparities + d]

                L_r0_last[d] = cost
                L_r1_last[j1*disp2 + d] = cost
                L_r2_last[j1*disp2 + d] = cost
                L_r3_last[j1*disp2 + d] = cost

                if cost < minCost:
                    minCost = cost
                S[i1 * width * numDisparities + j1 * numDisparities + d] += cost

        # 存储最低代价
        minL_r0_last[0] = minCost
        minL_r1_last[0] = minCost
        minL_r2_last[0] = minCost
        minL_r3_last[0] = minCost

        # 第一行剩余像素
        for j in range(j1+dj, j2, dj):
            minCost = MAX_SGM_COST
            minL_r0[0] = MAX_SGM_COST

            for d in range(numDisparities):
                cost = dsi[i1*width*numDisparities + j*numDisparities + d]

                L_r1_last[j*disp2 + d] = cost
                L_r2_last[j*disp2 + d] = cost
                L_r3_last[j*disp2 + d] = cost

                if cost < minCost:
                    minCost = cost

                # 计算L_r0上的最低代价
                mincostL = L_r0_last[d]           # 原始代价：C(p, d)
                costp11  = L_r0_last[d - 1] + P1  # P1代价：L_r(p-r, d-1) + P1
                costp12  = L_r0_last[d + 1] + P1  # P1低价：L_r(p-r, d+1) + P1
                costp2   = minL_r0_last[0] + P2   # P2代价：min L_r(p-r, k) + P2
                if mincostL > costp11:
                    mincostL = costp11
                if mincostL > costp12:
                    mincostL = costp12
                if mincostL > costp2:
                    mincostL = costp2
                delta_mincost = mincostL - minL_r0_last[0]
                newCost = saturate_cast_uint16(cost + delta_mincost)
                L_r0[d] = newCost

                if minL_r0[0] > newCost:
                    minL_r0[0] = newCost

                if pass_t == 0:
                    S[i1*width*numDisparities + j*numDisparities + d] = newCost
                else:
                    S[i1*width*numDisparities + j*numDisparities + d] += newCost

            minL_r1_last[j] = minCost
            minL_r2_last[j] = minCost
            minL_r3_last[j] = minCost

            # 交换L0的缓存
            swapPointer(&L_r0, &L_r0_last)
            swapPointer(&minL_r0, &minL_r0_last)

            # 边界赋值
            L_r1[j*disp2 - 1] = L_r2[j*disp2 - 1] = L_r3[j*disp2 - 1] = MAX_SGM_COST
            memset(&L_r1[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*(8-1))
            memset(&L_r2[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*(8-1))
            memset(&L_r3[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*(8-1))

            L_r1_last[j*disp2 - 1] = L_r2_last[j*disp2 - 1] = L_r3_last[j*disp2 - 1] = MAX_SGM_COST
            memset(&L_r1_last[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*(8-1))
            memset(&L_r2_last[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*(8-1))
            memset(&L_r3_last[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*(8-1))


        # 剩余行像素聚合
        for i in range(i1+di, i2, di):
            memset(L_r0_last, 0, sizeof(uint16_t)* numDisparities)
            minL_r0_last[0] = 0

            for j in range(j1, j2, dj):
                # 初始化
                minLr_08 = _mm_set1_epi16(MAX_SGM_COST)
                minLr_18 = _mm_set1_epi16(MAX_SGM_COST)
                minLr_28 = _mm_set1_epi16(MAX_SGM_COST)
                minLr_38 = _mm_set1_epi16(MAX_SGM_COST)

                minLr_last_08 = _mm_set1_epi16(minL_r0_last[0])
                minLr_last_18 = _mm_set1_epi16(minL_r1_last[j-dj])
                minLr_last_28 = _mm_set1_epi16(minL_r2_last[j])
                minLr_last_38 = _mm_set1_epi16(minL_r3_last[j+dj])

                # P2代价
                costp28_r0 = _mm_adds_epu16(minLr_last_08, P2_8)
                costp28_r1 = _mm_adds_epu16(minLr_last_18, P2_8)
                costp28_r2 = _mm_adds_epu16(minLr_last_28, P2_8)
                costp28_r3 = _mm_adds_epu16(minLr_last_38, P2_8)

                baseindexr0 = 0
                baseindexr1 = (j-dj)*disp2
                baseindexr2 = j*disp2
                baseindexr3 = (j+dj)*disp2

                # 一次性处理8个视差下的聚合代价
                upper8_r0 = _mm_load_si128(<__m128i*> (L_r0_last + baseindexr0 - 1))
                upper8_r1 = _mm_load_si128(<__m128i*> (L_r1_last + baseindexr1 - 1))
                upper8_r2 = _mm_load_si128(<__m128i*> (L_r2_last + baseindexr2 - 1))
                upper8_r3 = _mm_load_si128(<__m128i*> (L_r3_last + baseindexr3 - 1))

                for d in range(0, numDisparities, 8):
                    newCost8     = _mm_setzero_si128()
                    newCost8_all = _mm_setzero_si128()
                    cost8 = _mm_load_si128(<__m128i*> (dsi + i*width*numDisparities + j*numDisparities + d))

                    # L_r0方向代价聚合
                    baseindexr0 = d - 1
                    lower8_r0 = upper8_r0
                    upper8_r0 = _mm_load_si128(<__m128i*> (L_r0_last + baseindexr0 + 8))
                    # P1代价
                    costpm8_r0 = _mm_adds_epu16(lower8_r0, P1_8)
                    costpp8_r0 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r0, lower8_r0, 4), P1_8)
                    # 原始代价
                    mincost8 = _mm_alignr_epi8(upper8_r0, lower8_r0, 2)
                    # 聚合
                    mincost8 = _mm_min_epu16(mincost8, costpm8_r0)
                    mincost8 = _mm_min_epu16(mincost8, costpp8_r0)
                    mincost8 = _mm_min_epu16(mincost8, costp28_r0)
                    mincost8 = _mm_subs_epu16(mincost8, minLr_last_08)
                    newCost8 = _mm_adds_epu16(cost8, mincost8)

                    # 存储L_r0路径当前8个视差下的聚合代价
                    _mm_storeu_si128(<__m128i*> (L_r0 + d), newCost8)
                    minLr_08 = _mm_min_epu16(minLr_08, newCost8)

                    newCost8_all = _mm_adds_epu16(newCost8_all, newCost8)

                    # L_r1方向代价聚合
                    baseindexr1 = (j-dj)*disp2 + d - 1
                    lower8_r1 = upper8_r1
                    upper8_r1 = _mm_load_si128(<__m128i*> (L_r1_last + baseindexr1 + 8))
                    # P1代价
                    costpm8_r1 = _mm_adds_epu16(lower8_r1, P1_8)
                    costpp8_r1 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r1, lower8_r1, 4), P1_8)
                    # 原始代价
                    mincost8 = _mm_alignr_epi8(upper8_r1, lower8_r1, 2)
                    # 聚合
                    mincost8 = _mm_min_epu16(mincost8, costpm8_r1)
                    mincost8 = _mm_min_epu16(mincost8, costpp8_r1)
                    mincost8 = _mm_min_epu16(mincost8, costp28_r1)
                    mincost8 = _mm_subs_epu16(mincost8, minLr_last_18)
                    newCost8 = _mm_adds_epu16(cost8, mincost8)

                    # 存储L_r1路径当前8个视差下的聚合代价
                    _mm_storeu_si128(<__m128i*> (L_r1 + j*disp2 + d), newCost8)
                    minLr_18 = _mm_min_epu16(minLr_18, newCost8)

                    newCost8_all = _mm_adds_epu16(newCost8_all, newCost8)

                    # L_r2方向代价聚合
                    baseindexr2 = j * disp2 + d - 1
                    lower8_r2 = upper8_r2
                    upper8_r2 = _mm_load_si128(<__m128i*> (L_r2_last + baseindexr2 + 8))
                    # P1代价
                    costpm8_r2 = _mm_adds_epu16(lower8_r2, P1_8)
                    costpp8_r2 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r2, lower8_r2, 4), P1_8)
                    # 原始代价
                    mincost8 = _mm_alignr_epi8(upper8_r2, lower8_r2, 2)
                    # 聚合
                    mincost8 = _mm_min_epu16(mincost8, costpm8_r2)
                    mincost8 = _mm_min_epu16(mincost8, costpp8_r2)
                    mincost8 = _mm_min_epu16(mincost8, costp28_r2)
                    mincost8 = _mm_subs_epu16(mincost8, minLr_last_28)
                    newCost8 = _mm_adds_epu16(cost8, mincost8)

                    # 存储L_r2路径当前8个视差下的聚合代价
                    _mm_storeu_si128(<__m128i*> (L_r2 + j*disp2 + d), newCost8)
                    minLr_28 = _mm_min_epu16(minLr_28, newCost8)

                    newCost8_all = _mm_adds_epu16(newCost8_all, newCost8)

                    # L_r3方向代价聚合
                    baseindexr3 = (j+dj) * disp2 + d - 1
                    lower8_r3 = upper8_r3
                    upper8_r3 = _mm_load_si128(<__m128i*> (L_r3_last + baseindexr3 + 8))
                    # P1代价
                    costpm8_r3 = _mm_adds_epu16(lower8_r3, P1_8)
                    costpp8_r3 = _mm_adds_epu16(_mm_alignr_epi8(upper8_r3, lower8_r3, 4), P1_8)
                    # 原始代价
                    mincost8 = _mm_alignr_epi8(upper8_r3, lower8_r3, 2)
                    # 聚合
                    mincost8 = _mm_min_epu16(mincost8, costpm8_r3)
                    mincost8 = _mm_min_epu16(mincost8, costpp8_r3)
                    mincost8 = _mm_min_epu16(mincost8, costp28_r3)
                    mincost8 = _mm_subs_epu16(mincost8, minLr_last_38)
                    newCost8 = _mm_adds_epu16(cost8, mincost8)

                    # 存储L_r3路径当前8个视差下的聚合代价
                    _mm_storeu_si128(<__m128i*> (L_r3 + j*disp2 + d), newCost8)
                    minLr_38 = _mm_min_epu16(minLr_38, newCost8)

                    newCost8_all = _mm_adds_epu16(newCost8_all, newCost8)

                    if pass_t == 0:
                        S[i*width*numDisparities + j*numDisparities + d] = <uint16_t>_mm_extract_epi16(newCost8_all, 0)
                        if d + 1 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 1] = <uint16_t>_mm_extract_epi16(newCost8_all, 1)
                        if d + 2 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 2] = <uint16_t>_mm_extract_epi16(newCost8_all, 2)
                        if d + 3 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 3] = <uint16_t>_mm_extract_epi16(newCost8_all, 3)
                        if d + 4 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 4] = <uint16_t>_mm_extract_epi16(newCost8_all, 4)
                        if d + 5 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 5] = <uint16_t>_mm_extract_epi16(newCost8_all, 5)
                        if d + 6 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 6] = <uint16_t>_mm_extract_epi16(newCost8_all, 6)
                        if d + 7 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 7] = <uint16_t>_mm_extract_epi16(newCost8_all, 7)
                    else:
                        S[i*width*numDisparities + j*numDisparities + d] += <uint16_t>_mm_extract_epi16(newCost8_all, 0)
                        if d + 1 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 1] += <uint16_t>_mm_extract_epi16(newCost8_all, 1)
                        if d + 2 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 2] += <uint16_t>_mm_extract_epi16(newCost8_all, 2)
                        if d + 3 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 3] += <uint16_t>_mm_extract_epi16(newCost8_all, 3)
                        if d + 4 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 4] += <uint16_t>_mm_extract_epi16(newCost8_all, 4)
                        if d + 5 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 5] += <uint16_t>_mm_extract_epi16(newCost8_all, 5)
                        if d + 6 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 6] += <uint16_t>_mm_extract_epi16(newCost8_all, 6)
                        if d + 7 < numDisparities:
                            S[i*width*numDisparities + j*numDisparities + d + 7] += <uint16_t>_mm_extract_epi16(newCost8_all, 7)

                minL_r0[0] = <uint16_t> _mm_extract_epi16(_mm_minpos_epu16(minLr_08), 0)
                minL_r1[j] = <uint16_t> _mm_extract_epi16(_mm_minpos_epu16(minLr_18), 0)
                minL_r2[j] = <uint16_t> _mm_extract_epi16(_mm_minpos_epu16(minLr_28), 0)
                minL_r3[j] = <uint16_t> _mm_extract_epi16(_mm_minpos_epu16(minLr_38), 0)

                swapPointer(&L_r0, &L_r0_last)
                swapPointer(&minL_r0, &minL_r0_last)

            swapPointer(&L_r1, &L_r1_last)
            swapPointer(&L_r2, &L_r2_last)
            swapPointer(&L_r3, &L_r3_last)
            swapPointer(&minL_r1, &minL_r1_last)
            swapPointer(&minL_r2, &minL_r2_last)
            swapPointer(&minL_r3, &minL_r3_last)


    _mm_free(L_r0 - 1)
    _mm_free(L_r0_last - 1)
    _mm_free(L_r1 - disp2 - 1)
    _mm_free(L_r1_last - disp2 - 1)
    _mm_free(L_r2 - 1)
    _mm_free(L_r2_last - 1)
    _mm_free(L_r3 - disp2 - 1)
    _mm_free(L_r3_last - disp2 - 1)
    _mm_free(minL_r1 - 1)
    _mm_free(minL_r1_last - 1)
    _mm_free(minL_r2)
    _mm_free(minL_r2_last)
    _mm_free(minL_r3 - 1)
    _mm_free(minL_r3_last - 1)



# Horizontally compute the minimum amongst the packed unsigned 16-bit integers in a
# and store the minimum and index in dst, and zero the remaining bits in dst.
cdef inline __m128i _mm256_minpos_epu16(__m256i a):
    cdef __m128i high, low
    cdef __m128i min
    low  = _mm256_extractf128_si256(a, 0)
    high = _mm256_extractf128_si256(a, 1)
    min  = _mm_min_epu16(low, high)

    return _mm_minpos_epu16(min)



cdef void aggregate_costs_AVX(uint16_t* dsi, uint16_t* S, int width, int height, int P1, int P2, int numDisparities):
    # 支持avx
    cdef int width2 = width + 2
    cdef int disp2  = numDisparities + (256 / 16) + 1
    cdef uint16_t MAX_SGM_COST = UINT16_MAX
    cdef int Passes = 2

    # 路径代价
    cdef uint16_t* L_r0      = <uint16_t*>_mm_malloc(disp2 * sizeof(uint16_t), 16) + 1
    cdef uint16_t* L_r0_last = <uint16_t*>_mm_malloc(disp2 * sizeof(uint16_t), 16) + 1
    cdef uint16_t* L_r1      = <uint16_t*>_mm_malloc(width2 * disp2 * sizeof(uint16_t), 16) + disp2 + 1
    cdef uint16_t* L_r1_last = <uint16_t*>_mm_malloc(width2 * disp2 * sizeof(uint16_t), 16) + disp2 + 1
    cdef uint16_t* L_r2      = <uint16_t*>_mm_malloc(width * disp2 * sizeof(uint16_t), 16) + 1
    cdef uint16_t* L_r2_last = <uint16_t*>_mm_malloc(width * disp2 * sizeof(uint16_t), 16) + 1
    cdef uint16_t* L_r3      = <uint16_t*>_mm_malloc(width2 * disp2 * sizeof(uint16_t), 16) + disp2 + 1
    cdef uint16_t* L_r3_last = <uint16_t*>_mm_malloc(width2 * disp2 * sizeof(uint16_t), 16) + disp2 + 1

    # 最低代价
    cdef uint16_t minL_r0_array[2]
    cdef uint16_t* minL_r0      = &minL_r0_array[0]
    cdef uint16_t* minL_r0_last = &minL_r0_array[1]
    cdef uint16_t* minL_r1      = <uint16_t*>_mm_malloc(width2*sizeof(uint16_t), 16) + 1
    cdef uint16_t* minL_r1_last = <uint16_t*>_mm_malloc(width2*sizeof(uint16_t), 16) + 1
    cdef uint16_t* minL_r2      = <uint16_t*>_mm_malloc(width*sizeof(uint16_t), 16)
    cdef uint16_t* minL_r2_last = <uint16_t*>_mm_malloc(width*sizeof(uint16_t), 16)
    cdef uint16_t* minL_r3      = <uint16_t*>_mm_malloc(width2*sizeof(uint16_t), 16) + 1
    cdef uint16_t* minL_r3_last = <uint16_t*>_mm_malloc(width2*sizeof(uint16_t), 16) + 1

    # 边界赋值
    memset(&L_r1[-disp2], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r1_last[-disp2], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r3[-disp2], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r3_last[-disp2], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    L_r1[-disp2-1] = L_r1_last[-disp2-1] = MAX_SGM_COST
    L_r3[-disp2-1] = L_r3_last[-disp2-1] = MAX_SGM_COST

    memset(&L_r1[width * disp2 - 1], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r1_last[width * disp2 - 1], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r3[width * disp2 - 1], MAX_SGM_COST, sizeof(uint16_t)*disp2)
    memset(&L_r3_last[width * disp2 - 1], MAX_SGM_COST, sizeof(uint16_t)*disp2)


    L_r0[-1] = L_r2[-1]= MAX_SGM_COST
    L_r0_last[-1] = L_r2_last[-1] = MAX_SGM_COST
    memset(&L_r0[numDisparities], MAX_SGM_COST, sizeof(uint16_t) * 16)
    memset(&L_r2[numDisparities], MAX_SGM_COST, sizeof(uint16_t) * 16)
    memset(&L_r0_last[numDisparities], MAX_SGM_COST, sizeof(uint16_t) * 16)
    memset(&L_r2_last[numDisparities], MAX_SGM_COST, sizeof(uint16_t) * 16)

    minL_r1[-1] = minL_r1_last[-1] = 0
    minL_r1[width] = minL_r1_last[width] = 0
    minL_r3[-1] = minL_r3_last[-1] = 0
    minL_r3[width] = minL_r3_last[width] = 0

    cdef Py_ssize_t pass_t
    cdef Py_ssize_t i, j, d
    cdef Py_ssize_t i1, i2, di
    cdef Py_ssize_t j1, j2, dj
    cdef uint16_t cost, minCost, newCost, newCost_all
    cdef int mincostL, costp11, costp12, costp2, delta_mincost

    cdef int baseindexr0, baseindexr1, baseindexr2, baseindexr3
    cdef __m256i minLr_016, minLr_116, minLr_216, minLr_316
    cdef __m256i minLr_last_016, minLr_last_116, minLr_last_216, minLr_last_316
    cdef __m256i costpm16_r0, costpm16_r1, costpm16_r2, costpm16_r3
    cdef __m256i costpp16_r0, costpp16_r1, costpp16_r2, costpp16_r3
    cdef __m256i costp216_r0, costp216_r1, costp216_r2, costp216_r3
    cdef __m256i cost16, mincost16, newCost16, newCost16_all
    cdef __m256i P1_16 = _mm256_set1_epi16(<uint16_t> P1)
    cdef __m256i P2_16 = _mm256_set1_epi16(<uint16_t> P2)

    for pass_t in range(Passes):
        if pass_t == 0:
            i1, i2, di = 0, height, 1
            j1, j2, dj = 0, width,  1
        else:
            i1, i2, di = height-1, -1, -1
            j1, j2, dj = width-1, -1, -1

        # 第一行第一个像素
        minCost = MAX_SGM_COST
        if pass_t == 0:
            for d in range(numDisparities):
                cost = dsi[i1*width*numDisparities + j1*numDisparities + d]

                L_r0_last[d] = cost
                L_r1_last[j1*disp2 + d] = cost
                L_r2_last[j1*disp2 + d] = cost
                L_r3_last[j1*disp2 + d] = cost

                if cost < minCost:
                    minCost = cost
                S[i1 * width * numDisparities + j1 * numDisparities + d] = cost
        else:
            for d in range(numDisparities):
                cost = dsi[i1*width*numDisparities + j1*numDisparities + d]

                L_r0_last[d] = cost
                L_r1_last[j1*disp2 + d] = cost
                L_r2_last[j1*disp2 + d] = cost
                L_r3_last[j1*disp2 + d] = cost

                if cost < minCost:
                    minCost = cost
                S[i1 * width * numDisparities + j1 * numDisparities + d] += cost

        # 存储最低代价
        minL_r0_last[0] = minCost
        minL_r1_last[0] = minCost
        minL_r2_last[0] = minCost
        minL_r3_last[0] = minCost

        # 第一行剩余像素
        for j in range(j1+dj, j2, dj):
            minCost = MAX_SGM_COST
            minL_r0[0] = MAX_SGM_COST

            for d in range(numDisparities):
                cost = dsi[i1*width*numDisparities + j*numDisparities + d]

                L_r1_last[j*disp2 + d] = cost
                L_r2_last[j*disp2 + d] = cost
                L_r3_last[j*disp2 + d] = cost

                if cost < minCost:
                    minCost = cost

                # 计算L_r0上的最低代价
                mincostL = L_r0_last[d]           # 原始代价：C(p, d)
                costp11  = L_r0_last[d - 1] + P1  # P1代价：L_r(p-r, d-1) + P1
                costp12  = L_r0_last[d + 1] + P1  # P1低价：L_r(p-r, d+1) + P1
                costp2   = minL_r0_last[0] + P2   # P2代价：min L_r(p-r, k) + P2
                if mincostL > costp11:
                    mincostL = costp11
                if mincostL > costp12:
                    mincostL = costp12
                if mincostL > costp2:
                    mincostL = costp2
                delta_mincost = mincostL - minL_r0_last[0]
                newCost = saturate_cast_uint16(cost + delta_mincost)
                L_r0[d] = newCost

                if minL_r0[0] > newCost:
                    minL_r0[0] = newCost

                if pass_t == 0:
                    S[i1*width*numDisparities + j*numDisparities + d] = newCost
                else:
                    S[i1*width*numDisparities + j*numDisparities + d] += newCost

            minL_r1_last[j] = minCost
            minL_r2_last[j] = minCost
            minL_r3_last[j] = minCost

            # 交换L0的缓存
            swapPointer(&L_r0, &L_r0_last)
            swapPointer(&minL_r0, &minL_r0_last)

            # 边界赋值
            L_r1[j*disp2 - 1] = L_r2[j*disp2 - 1] = L_r3[j*disp2 - 1] = MAX_SGM_COST
            memset(&L_r1[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*16)
            memset(&L_r2[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*16)
            memset(&L_r3[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*16)

            L_r1_last[j*disp2 - 1] = L_r2_last[j*disp2 - 1] = L_r3_last[j*disp2 - 1] = MAX_SGM_COST
            memset(&L_r1_last[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*16)
            memset(&L_r2_last[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*16)
            memset(&L_r3_last[j*disp2 + numDisparities], MAX_SGM_COST, sizeof(uint16_t)*16)


        # 剩余行像素聚合
        for i in range(i1+di, i2, di):
            memset(L_r0_last, 0, sizeof(uint16_t)* numDisparities)
            minL_r0_last[0] = 0

            for j in range(j1, j2, dj):
                # 初始化
                minLr_016 = _mm256_set1_epi16(MAX_SGM_COST)
                minLr_116 = _mm256_set1_epi16(MAX_SGM_COST)
                minLr_216 = _mm256_set1_epi16(MAX_SGM_COST)
                minLr_316 = _mm256_set1_epi16(MAX_SGM_COST)

                minLr_last_016 = _mm256_set1_epi16(minL_r0_last[0])
                minLr_last_116 = _mm256_set1_epi16(minL_r1_last[j-dj])
                minLr_last_216 = _mm256_set1_epi16(minL_r2_last[j])
                minLr_last_316 = _mm256_set1_epi16(minL_r3_last[j+dj])

                # P2代价
                costp216_r0 = _mm256_adds_epu16(minLr_last_016, P2_16)
                costp216_r1 = _mm256_adds_epu16(minLr_last_116, P2_16)
                costp216_r2 = _mm256_adds_epu16(minLr_last_216, P2_16)
                costp216_r3 = _mm256_adds_epu16(minLr_last_316, P2_16)

                # 一次性处理16个视差下的聚合代价
                for d in range(0, numDisparities, 16):
                    newCost16     = _mm256_setzero_si256()
                    newCost16_all = _mm256_setzero_si256()
                    cost16 = _mm256_load_si256(<__m256i*> (dsi + i*width*numDisparities + j*numDisparities + d))

                    # L_r0方向代价聚合
                    # P1代价
                    baseindexr0 = d - 1
                    costpm16_r0 = _mm256_adds_epu16(_mm256_load_si256(<__m256i*> (L_r0_last + baseindexr0)), P1_16)
                    costpp16_r0 = _mm256_adds_epu16(_mm256_load_si256(<__m256i*> (L_r0_last + baseindexr0 + 2)), P1_16)
                    # 原始代价
                    mincost16 = _mm256_load_si256(<__m256i*> (L_r0_last + baseindexr0 + 1))
                    # 聚合
                    mincost16 = _mm256_min_epu16(mincost16, costpm16_r0)
                    mincost16 = _mm256_min_epu16(mincost16, costpp16_r0)
                    mincost16 = _mm256_min_epu16(mincost16, costp216_r0)
                    mincost16 = _mm256_subs_epu16(mincost16, minLr_last_016)
                    newCost16 = _mm256_adds_epu16(cost16, mincost16)

                    # 存储L_r0路径当前16个视差下的聚合代价
                    _mm256_storeu_si256(<__m256i*> (L_r0 + d), newCost16)
                    minLr_016 = _mm256_min_epu16(minLr_016, newCost16)

                    newCost16_all = _mm256_adds_epu16(newCost16_all, newCost16)

                    # L_r1方向代价聚合
                    # P1代价
                    baseindexr1 = (j-dj)*disp2 + d - 1
                    costpm16_r1 = _mm256_adds_epu16(_mm256_load_si256(<__m256i*> (L_r1_last + baseindexr1)), P1_16)
                    costpp16_r1 = _mm256_adds_epu16(_mm256_load_si256(<__m256i*> (L_r1_last + baseindexr1 + 2)), P1_16)
                    # 原始代价
                    mincost16 = _mm256_load_si256(<__m256i*> (L_r1_last + baseindexr1 + 1))
                    # 聚合
                    mincost16 = _mm256_min_epu16(mincost16, costpm16_r1)
                    mincost16 = _mm256_min_epu16(mincost16, costpp16_r1)
                    mincost16 = _mm256_min_epu16(mincost16, costp216_r1)
                    mincost16 = _mm256_subs_epu16(mincost16, minLr_last_116)
                    newCost16 = _mm256_adds_epu16(cost16, mincost16)

                    # 存储L_r1路径当前16个视差下的聚合代价
                    _mm256_storeu_si256(<__m256i*> (L_r1 + j*disp2 + d), newCost16)
                    minLr_116 = _mm256_min_epu16(minLr_116, newCost16)

                    newCost16_all = _mm256_adds_epu16(newCost16_all, newCost16)

                    # L_r2方向代价聚合
                    # P1代价
                    baseindexr2 = j * disp2 + d - 1
                    costpm16_r2 = _mm256_adds_epu16(_mm256_load_si256(<__m256i*> (L_r2_last + baseindexr2)), P1_16)
                    costpp16_r2 = _mm256_adds_epu16(_mm256_load_si256(<__m256i*> (L_r2_last + baseindexr2 + 2)), P1_16)
                    # 原始代价
                    mincost16 = _mm256_load_si256(<__m256i*> (L_r2_last + baseindexr2 + 1))
                    # 聚合
                    mincost16 = _mm256_min_epu16(mincost16, costpm16_r2)
                    mincost16 = _mm256_min_epu16(mincost16, costpp16_r2)
                    mincost16 = _mm256_min_epu16(mincost16, costp216_r2)
                    mincost16 = _mm256_subs_epu16(mincost16, minLr_last_216)
                    newCost16 = _mm256_adds_epu16(cost16, mincost16)

                    # 存储L_r2路径当前16个视差下的聚合代价
                    _mm256_storeu_si256(<__m256i*> (L_r2 + j*disp2 + d), newCost16)
                    minLr_216 = _mm256_min_epu16(minLr_216, newCost16)

                    newCost16_all = _mm256_adds_epu16(newCost16_all, newCost16)

                    # L_r3方向代价聚合
                    # P1代价
                    baseindexr3 = (j+dj) * disp2 + d - 1
                    costpm16_r3 = _mm256_adds_epu16(_mm256_load_si256(<__m256i*> (L_r3_last + baseindexr3)), P1_16)
                    costpp16_r3 = _mm256_adds_epu16(_mm256_load_si256(<__m256i*> (L_r3_last + baseindexr3 + 2)), P1_16)
                    # 原始代价
                    mincost16 = _mm256_load_si256(<__m256i*> (L_r3_last + baseindexr3 + 1))
                    # 聚合
                    mincost16 = _mm256_min_epu16(mincost16, costpm16_r3)
                    mincost16 = _mm256_min_epu16(mincost16, costpp16_r3)
                    mincost16 = _mm256_min_epu16(mincost16, costp216_r3)
                    mincost16 = _mm256_subs_epu16(mincost16, minLr_last_316)
                    newCost16 = _mm256_adds_epu16(cost16, mincost16)

                    # 存储L_r3路径当前16个视差下的聚合代价
                    _mm256_storeu_si256(<__m256i*> (L_r3 + j*disp2 + d), newCost16)
                    minLr_316 = _mm256_min_epu16(minLr_316, newCost16)

                    newCost16_all = _mm256_adds_epu16(newCost16_all, newCost16)

                    if pass_t == 0:
                        _mm256_storeu_si256(<__m256i*>(S+i*width*numDisparities + j*numDisparities + d), newCost16_all)
                    else:
                        _mm256_storeu_si256(<__m256i*>(S+i*width*numDisparities + j*numDisparities + d), _mm256_adds_epu16(_mm256_loadu_si256(<__m256i*>(S+i*width*numDisparities + j*numDisparities + d)), newCost16_all))

                minL_r0[0] = <uint16_t> _mm_extract_epi16(_mm256_minpos_epu16(minLr_016), 0)
                minL_r1[j] = <uint16_t> _mm_extract_epi16(_mm256_minpos_epu16(minLr_116), 0)
                minL_r2[j] = <uint16_t> _mm_extract_epi16(_mm256_minpos_epu16(minLr_216), 0)
                minL_r3[j] = <uint16_t> _mm_extract_epi16(_mm256_minpos_epu16(minLr_316), 0)

                swapPointer(&L_r0, &L_r0_last)
                swapPointer(&minL_r0, &minL_r0_last)

            swapPointer(&L_r1, &L_r1_last)
            swapPointer(&L_r2, &L_r2_last)
            swapPointer(&L_r3, &L_r3_last)
            swapPointer(&minL_r1, &minL_r1_last)
            swapPointer(&minL_r2, &minL_r2_last)
            swapPointer(&minL_r3, &minL_r3_last)


    _mm_free(L_r0 - 1)
    _mm_free(L_r0_last - 1)
    _mm_free(L_r1 - disp2 - 1)
    _mm_free(L_r1_last - disp2 - 1)
    _mm_free(L_r2 - 1)
    _mm_free(L_r2_last - 1)
    _mm_free(L_r3 - disp2 - 1)
    _mm_free(L_r3_last - disp2 - 1)
    _mm_free(minL_r1 - 1)
    _mm_free(minL_r1_last - 1)
    _mm_free(minL_r2)
    _mm_free(minL_r2_last)
    _mm_free(minL_r3 - 1)
    _mm_free(minL_r3_last - 1)



print('可用处理器数：', openmp.omp_get_num_procs())























