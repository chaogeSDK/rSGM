# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import time
import rSGM


if __name__ == '__main__':
    left = cv2.imread(path1, -1)
    righ = cv2.imread(path2, -1)
    print("输入图像尺寸：", left.shape[1], 'x', left.shape[0])
    # left = cv2.pyrDown(left)
    # righ = cv2.pyrDown(righ)

    # 彩色图-->灰度图
    if left.ndim >= 3:
        iml = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    else:
        iml = left
    if righ.ndim >= 3:
        imr = cv2.cvtColor(righ, cv2.COLOR_BGR2GRAY)
    else:
        imr = righ

    # 去噪
    # iml = cv2.fastNlMeansDenoising(iml, h=10, templateWindowSize=7, searchWindowSize=21)
    # imr = cv2.fastNlMeansDenoising(imr, h=10, templateWindowSize=7, searchWindowSize=21)


    # SGM参数设置
    param = rSGM.stereoParametersSGM(numDisparities=128, P1=20, P2=200, cheight = 7, cwidth = 9)
    param.setDescriptorType(3)
    param.setLineLength(8)
    param.setUniquenessRatio(0.85)
    param.setLRThreshold(1)
    param.setSpeckleRange(1)
    param.setSpeckleWindowSize(100)
    param.setLRCcheck(True)
    param.setSubRefine(True)
    param.setMedianFilter(True)
    param.setNLMFilter(False)

    # 匹配
    tic = time.time()
    rsgm = rSGM.stereoRSGM(iml.shape[1], iml.shape[0], param, numStrips = 0, border= 20)
    rsgm.compute(iml, imr)
    displ = rsgm.get_disparity_map_left()
    displ = (displ / 16.).astype(np.float32)
    dispr = rsgm.get_disparity_map_right()
    toc = time.time()
    print('\t用时{:.3f}s'.format(toc - tic))

    cv2.imwrite('/home/chaoshui/Downloads/displ.jpg', displ*3)
    cv2.imwrite('/home/chaoshui/Downloads/dispr.jpg', dispr*3)








