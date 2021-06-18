#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def lines_detector_hough(edge,ThetaDim = None,DistStep = None,threshold = None, halfThetaWindowSize = 2, halfDistWindowSize = None):
    '''
    :param edge: 经过边缘检测得到的二值图
    :param ThetaDim: hough空间中theta轴的刻度数量(将[0,pi)均分为多少份),反应theta轴的粒度,越大粒度越细，也就是横轴的箱子数目
    :param DistStep: hough空间中dist轴的划分粒度,即dist轴的最小单位长度，纵向的格子数目
    :param threshold: 投票表决认定存在直线的起始阈值
    :return: 返回检测出的所有直线的参数(theta,dist)
    '''
    imgsize = edge.shape
    if ThetaDim == None:
        ThetaDim = 90  # 360/90 =4 也就是一个格子4度，看做是每个州
    if DistStep == None:
        DistStep = 1
    # 计算图像对角线的长度，最大的r就是对角线的长度
    MaxDist = np.sqrt(imgsize[0]**2 + imgsize[1]**2)
    # 通过将r来除以step得到纵向的箱子数目， 看做是每个州的候选人
    DistDim = int(np.ceil(MaxDist/DistStep))

    # 用来控制极大值抑制的窗的大小
    if halfDistWindowSize == None:
        halfDistWindowSize = int(DistDim/50)

    # 将投票箱设置为0
    accumulator = np.zeros((ThetaDim, DistDim)) # theta的范围是[0,pi). 在这里将[0,pi)进行了线性映射.类似的,也对Dist轴进行了线性映射

    # 将所有的θ对应的cos和sin的值全部计算出来[sin0, sin4, sin8.....sin360]
    sinTheta = [np.sin(t*np.pi/ThetaDim) for t in range(ThetaDim)]
    cosTheta = [np.cos(t*np.pi/ThetaDim) for t in range(ThetaDim)]

    # 遍历每个像素位置
    for i in range(imgsize[0]):
        for j in range(imgsize[1]):
            # 如果是边缘点
            if not edge[i, j] == 0:
                # 遍历每个角度，对每个角度都求出r，然后往纵向的箱子里投票
                # 也就是遍历每个州，利用直线函数的限制来计算出要投票的参数，看看参数落在哪个候选人的区间中就对他投一票
                for k in range(ThetaDim):
                    accumulator[k][int(round((i*cosTheta[k]+j*sinTheta[k])*DistDim/MaxDist))] += 1

    # 设置阈值，票数大于阈值的，都当做是直线
    M = accumulator.max()
    if threshold == None:
        threshold = int(M*2.3875/10)

    # 将所有大于阈值的投票箱取出来
    # np.where(condition) :取出所有满足条件的点的坐标（X，Y），其中X为array, Y也为array， 转换成[2, n]的坐标值
    #[2, n] => [n, 2]
    x, y = np.where(accumulator > threshold)
    result = np.array([[x[i], y[i]] for i in range(x.shape[0])])
    # result = np.array(np.where(accumulator > threshold))  # 阈值化
    # 非极大值抑制
    cood = [[], []]
    # 遍历所有满足条件的箱子
    for i in range(result.shape[0]):
        # 取出这些箱子的邻域的箱子
        eight_neiborhood = accumulator[max(0, result[i, 0] - halfThetaWindowSize + 1): min(result[i, 0] + halfThetaWindowSize, accumulator.shape[0]),
                                       max(0, result[i, 1] - halfDistWindowSize + 1) : min(result[i, 1] + halfDistWindowSize, accumulator.shape[1])]
        #
        if (accumulator[result[i, 0], result[i, 1]] >= eight_neiborhood).all():
            # 将横坐标添加到cood[0]中， 纵坐标添加到cood[1]中
            cood[0].append(result[i, 0])
            cood[1].append(result[i, 1])
    #[2, n1]
    res = np.array(cood)    # 非极大值抑制
    res = res.astype(np.float64)
    # [投票箱坐标乘以间隔值，得到真实的r和θ]
    res[0] = res[0]*np.pi/ThetaDim
    res[1] = res[1]*MaxDist/DistDim
    return res

def drawLines(lines, edge, color = (255,0,0),err = 3):
    if len(edge.shape) == 2:
        result = np.dstack((edge, edge, edge))
    else:
        result = edge
    # 将所有的参数求出来
    Cos = np.cos(lines[0])
    Sin = np.sin(lines[0])
    # 遍历图片所有的像素点， 阈值之内的点画红
    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            e = np.abs(lines[1] - i*Cos - j*Sin)
            if (e < err).any():
                result[i, j] = color
    return result


if __name__=='__main__':
    pic_path = './HoughImg/'
    pics = os.listdir(pic_path)

    for i in pics:
        if i.split('.')[-1] == 'jpeg' or i.split('.')[-1] == 'jpg':
            img = plt.imread(pic_path+i)
            # 高斯核卷积去噪
            blurred = cv2.GaussianBlur(img, (3, 3), 0)
            # plt.imshow(blurred, cmap='gray')
            # plt.axis('off')
            # plt.show()
            # 灰度化
            gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY) if not len(blurred.shape) == 2 else blurred
            # canary 边缘检测
            edge = cv2.Canny(gray, 50, 150)   #  二值图 (0 或 255) 得到 canny边缘检测的结果
            # 霍夫直线检测
            lines = lines_detector_hough(edge)
            # 画线函数
            final_img = drawLines(lines, blurred)

            plt.imshow(final_img, cmap='gray')
            plt.axis('off')
            plt.show()

