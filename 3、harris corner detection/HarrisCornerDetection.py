#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import math


def cal_conv_out_img_size(img_size, filter_size, padding=None, strides=1):
    '''
    计算卷积输出图像的尺寸
    '''
    if not padding:
        padding = [0]*4
    if not strides:
        strides = 1
    if len(filter_size) == 1:
        filter_size = [filter_size, filter_size]
    h = int(np.ceil((img_size[0] + padding[0] + padding[1] - filter_size[0])*1.0 / strides[0]) + 1)
    w = int(np.ceil((img_size[1] + padding[2] + padding[3] - filter_size[1])*1.0 / strides[1]) + 1)
    return [h, w]


def convolve(filter, mat, padding, strides, gray_flag=False):
    '''
    图像二维卷积
    :param filter:卷积核，必须为二维(2 x 1也算二维) 否则返回None
    :param mat:图片 [h,w] || [h,w,3]
    :param padding:对齐 [4] : [0,0,5,5]
    :param strides: int
    :return:返回卷积后的图片。(灰度图，彩图都适用)
    '''
    shape = cal_conv_out_img_size(mat.shape, filter.shape, padding, strides)
    result = np.zeros(shape)
    filter_size = filter.shape
    mat_size = mat.shape
    assert len(filter_size) == 2
    # 灰度图像
    if len(mat_size) == 2:
        mat = mat.reshape([mat_size[0], mat_size[1], 1])
        result = result.reshape(shape[0], shape[1], 1)
        mat_size = mat.shape
        gray_flag = True
    # 先遍历图像的行
    for c in range(mat_size[-1]):
        # pad: [0,0,5,5]   [h,w]=>[h, 5+w+5] :在图片的左右填充
        # pad: [5,5,0,0]   [h,w]=>[5+h+5, w] :在图片的上下填充
        pad_mat = np.pad(mat[:, :,  c], ((padding[0], padding[1]), (padding[2], padding[3])), 'constant')
        # 遍历图像的列
        for i in range(0, mat_size[0], strides[1]):
            # 以卷积核的左上角为原点，拿卷积核和图像的相应位置相乘然后相加，就是做卷积操作
            for j in range(0, mat_size[1], strides[0]):
                val = (filter*pad_mat[i:i+filter_size[0], j:j+filter_size[1]]).sum()
                result[i, j, c] = val
    return result.reshape([result.shape[0], result.shape[1]]) if gray_flag else result

def row_convolve(filter, mat, padding=None, strides=[1, 1]):
    '''
    线性卷积就是先用一维的卷积横向卷积
    :param filter:线性卷积核 [1, n]
    :param mat:图片 [h,w] || [h,w,3]
    :param padding:对齐
    :param strides:移动步长
    :return:返回卷积后的图片。(灰度图，彩图都适用) 若不是线性卷积核，返回None
    '''
    filter_size = filter.shape
    assert len(filter_size) <= 2
    if len(filter_size) == 1:
        filter = filter.reshape([1, -1])
        filter_size = filter.shape
    if padding == None or len(padding) < 2:
        padding = [filter_size[1]//2, filter_size[1]//2]
    result = convolve(filter, mat, [0, 0, padding[0], padding[1]], strides)
    return result


def col_convolve(filter, mat, padding=None, strides=[1, 1]):
    '''
    线性卷积就是先用一维的卷积横向卷积
    :param filter:线性卷积核 [n, 1]
    :param mat:图片 [h,w] || [h,w,3]
    :param padding:对齐
    :param strides:移动步长
    :return:返回卷积后的图片。(灰度图，彩图都适用) 若不是线性卷积核，返回None
    '''

    # 卷积核标准化为[n,1]
    filter_size = filter.shape
    assert len(filter_size) <= 2
    if len(filter_size) == 1:
        filter = filter.reshape([-1, 1])
        filter_size = filter.shape
    elif filter_size[0] == 1:
        filter = filter.reshape([-1, 1])
        filter_size = filter.shape
    if padding == None or len(padding) < 2:
        padding = [filter_size[0]//2, filter_size[0]//2]
    result = convolve(filter, mat, [padding[0], padding[1], 0, 0], strides)
    return result


def divided_convolve(filter, img):
    '''
    将二维卷积分解为横向线性卷积和纵向线性卷积的叠加，先横向线性卷积，再进行纵向线性卷积
    :param filter: 线性卷积核 touple: ([n], [n])  n为卷积核的大小
    :param mat: 图片 [h, w, 3]  ||  [h, w]
    :return: 卷积后的图片,(灰度图，彩图都适用)
    '''
    (row_filter, col_filter) = filter if len(filter) == 2 else (filter, filter)
    result = row_convolve(row_filter, img)
    result = col_convolve(col_filter, result)
    return result




def score_for_each_pixel(sq_img_gx,sq_img_gy,img_gx_gy,k):
    '''
    所传入的参数都只能有一个通道，且形状必须相同
    :param sq_img_gx: x方向上的梯度平方的图片
    :param sq_img_gy: y方向上的梯度平方的图片
    :param img_gx_gy: x,y方向上梯度乘积的图片
    :param k: 矩阵的迹前面的系数
    :return: 各点的得分
    '''
    result = []
    for i in range(sq_img_gx.shape[0]):
        result.append([])
        for j in range(sq_img_gx.shape[1]):
            # 每个像素都求一个M矩阵，计算所有像素的M矩阵
            M = np.array(
                [
                    [sq_img_gx[i,j], img_gx_gy[i,j]],
                    [img_gx_gy[i,j], sq_img_gy[i,j]]
                ]
            )
            # 计算所有像素点的 |M|-(K*trace(M)^2)
            result[-1].append(np.linalg.det(M)-k*(np.trace(M)**2))
    return np.array(result)

def Sign(img, score, area, decide_value=None, boder=[3,3,3,3]):
    '''
    :param img: 需要在角点处做标记的图片(可为多通道)
    :param score: 各个像素的角点得分
    :param area: 标记区域的大小(area[0] x area[1])
    :param decide_value: 决策是否为角点的阈值
    :param boder: 标记的边界宽度
    :return: 返回标记后的图片
    '''
    if decide_value == None:
        # 大于这个值的都认为是角点
        decide_value = 34*math.fabs(score.mean())  # 34这个参数可调
        print(decide_value)
    judger = score > decide_value
    final_img = img.copy()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 非极大值抑制
            isLocalExtreme = score[i, j] >= score[max(i-(area[0]//2),0):min(i+(area[0]//2)+1,img.shape[0]),max(j-(area[1]//2),0):min(j+(area[1]//2)+1,img.shape[1])] #非极值抑制
            # 如果该像素点求出来的判断值大于阈值，且该点为局部最大值，才对该点画小正方形
            if judger[i, j] and isLocalExtreme.all():
                for k in range(min(boder[0], area[1]//6+1)):
                    final_img[max(i-(area[0]//2),0):min(i+(area[0]//2)+1,img.shape[0]),max(j-(area[1]//2),0)+k,:] = [255,0,0]  #top
                for k in range(min(boder[1],area[1]//6+1)):
                    final_img[max(i-(area[0]//2),0):min(i+(area[0]//2)+1,img.shape[0]),min(j+(area[1]//2),img.shape[1]-1)-k,:] = [255,0,0] #bottom
                for k in range(min(boder[2],area[0]//6+1)):
                    final_img[max(i-(area[0]//2),0)+k,max(j-(area[1]//2),0):min(j+(area[1]//2)+1,img.shape[1]),:] = [255, 0, 0] #left
                for k in range(min(boder[3],area[0]//6+1)):
                    final_img[min(i+(area[0]//2),img.shape[0]-1)-k,max(j-(area[1]//2),0):min(j+(area[1]//2)+1,img.shape[1]),:] = [255,0,0]  # right
    return final_img

def OneDimensionStandardNormalDistribution(x,sigma):
    '''
    生成高斯核
    '''
    E = -0.5/(sigma*sigma)
    return 1/(math.sqrt(2*math.pi)*sigma)*math.exp(x*x*E)

if __name__ == '__main__':

    pic_path = './img/'
    pics = os.listdir(pic_path)

    # 高斯windows
    window = 1.0/159*np.array([
        [2,4,5,4,2],
        [4,9,12,9,4],
        [5,12,15,12,5],
        [4,9,12,9,4],
        [2,4,5,4,2]
    ])   # window(5x5 Gaussisan kernel)

    # 线性高斯
    linear_Gaussian_filter_5 = [2, 1, 0, 1, 2]
    sigma = 1.4
    linear_Gaussian_filter_5 = np.array([[OneDimensionStandardNormalDistribution(t, sigma) for t in linear_Gaussian_filter_5]])
    linear_Gaussian_filter_5 = linear_Gaussian_filter_5/linear_Gaussian_filter_5.sum()

    # sobel核
    G_y = np.array(
        [
            [2, 2, 4, 2, 2],
            [1, 1, 2, 1, 1],
            [0 ,0 ,0 ,0 ,0],
            [-1,-1,-2,-1,-1],
            [-2,-2,-4,-2,-2]
        ]
    )
    G_x = np.array(
        [
            [2, 1, 0, -1, -2],
            [2, 1, 0, -1, -2],
            [4, 2, 0, -2, -4],
            [2, 1, 0, -1, -2],
            [2, 1, 0, -1, -2]
        ]
    ) #5x5 sobel kernel

    for i in pics:
        if i[-5:] == '.jpeg':
            orignal_img = plt.imread(pic_path+i)

            plt.imshow(orignal_img)
            plt.axis('off')
            plt.show()

            img = orignal_img.mean(axis=-1)

            # 拿sobel核卷积相当于求了导数
            img_gx = convolve(G_x, img, [2,2,2,2], [1,1])
            img_gy = convolve(G_y, img, [2,2,2,2], [1,1])

            sq_img_gx = img_gx * img_gx
            sq_img_gy = img_gy * img_gy
            img_gx_gy = img_gx * img_gy

            # sq_img_gx = convolve(window, sq_img_gx, [2, 2, 2, 2], [1, 1])
            # sq_img_gy = convolve(window, sq_img_gy, [2, 2, 2, 2], [1, 1])
            # img_gx_gy = convolve(window, img_gx_gy, [2, 2, 2, 2], [1, 1])

            # 用window对Ix平方， Iy平方 Ix乘以Iy的平方 卷积一下，后面需要用
            sq_img_gx = divided_convolve(linear_Gaussian_filter_5, sq_img_gx)
            sq_img_gy = divided_convolve(linear_Gaussian_filter_5, sq_img_gy)
            img_gx_gy = divided_convolve(linear_Gaussian_filter_5, img_gx_gy)

            score = score_for_each_pixel(sq_img_gx, sq_img_gy, img_gx_gy, 0.05)
            # 画图
            final_img = Sign(orignal_img, score, [12, 12])

            plt.imshow(final_img.astype(np.uint8))
            plt.axis('off')
            plt.show()


