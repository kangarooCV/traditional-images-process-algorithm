#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import cv2

def cal_conv_out_img_size(img_size, filter_size, padding=None, strides=1):
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


def judgeConnect(m2, threshold):
    '''
    极大值过于稀疏，将极大值点的八邻域的模糊点也设为255，设置为边界，这样边界会更加明显
    :param m2:[h,w] 图像
    :param threshold: 阈值是上下限
    :return: 图像[h,w]
    '''

    e = 0.01
    # 存所有极大值点的坐标，假设有n个极大值点，最后s为[n, 2]
    s = []
    # 存所有点的坐标，遍历完后cood为[h,w,2]  2表示[x, y]
    cood = []
    # 遍历h
    for i in range(m2.shape[0]):
        cood.append([])
        # 遍历w
        for j in range(m2.shape[1]):
            cood[-1].append([i, j])
            if abs(m2[i, j] - 255) < e:
                s.append([i, j])
    cood = np.array(cood)
    # 如果栈没有空，就一直查找
    while not len(s) == 0:
        # 从栈中弹出一个值来判断其八邻域内是否有极大值点
        index = s.pop()
        # 得到index点的八邻域的像素值窗口
        jud = m2[max(0, index[0] - 1):min(index[0] + 2, m2.shape[1]), max(0, index[1] - 1):min(index[1] + 2, m2.shape[0])]
        # 取得index坐标的八邻域坐标点
        jud_i = cood[max(0, index[0] - 1):min(index[0] + 2, cood.shape[1]), max(0, index[1] - 1):min(index[1] + 2, cood.shape[0])]
        # 取出在八邻域内像素值在模糊区间的点的掩码mask
        jud = (jud > threshold[0]) & (jud < threshold[1])
        # 将这些点的坐标取出来
        jud_i = jud_i[jud]
        # 将这些点的坐标入栈，并将这些点的八领域的模糊点的像素置255，然后等待下一步继续判断，也就是将所有极大值点的八邻域的模糊点也作为边界
        for i in range(jud_i.shape[0]):
            s.append(list(jud_i[i]))
            m2[jud_i[i][0], jud_i[i][1]] = 255
    return m2


def DecideAndConnectEdge(g_l, g_t, threshold=None):
    '''
    非极大值抑制函数+连接函数
    :param g_l:图像每一点梯度的幅值 [h, w]
    :param g_t:图像每一点梯度的相角[h, w]
    :param threshold: 上下限
    :return:图片
    '''
    if threshold == None:
        lower_boundary = g_l.mean()*0.5
        threshold = [lower_boundary, lower_boundary*3]
    result = np.zeros(g_l.shape)
    for i in range(g_l.shape[0]):
        for j in range(g_l.shape[1]):
            isLocalExtreme = True
            # 得到一个像素点的八邻域，并限制不会超出图像的尺度
            eight_neiborhood = g_l[max(0, i-1): min(i+2, g_l.shape[0]), max(0, j-1): min(j+2, g_l.shape[1])]
            # 在图像内部可以双线性插值的点才进行非极大值抑制，图像边缘点直接设为极大值
            if eight_neiborhood.shape == (3, 3):
                # 梯度的正方向为dy向上，dx向左
                # 如果梯度的tanθ∈[0, 1], 角度在0到45度之间  abs(tanθ)=abs(dy/dx) = d/1 => d=abs(dy/dx)
                if 0 <= g_t[i, j] < 1:
                    d = abs(g_t[i, j])
                    first = eight_neiborhood[1, 2] + (eight_neiborhood[0, 2] - eight_neiborhood[1, 2]) * d
                    second = eight_neiborhood[1, 0] + (eight_neiborhood[2, 0] - eight_neiborhood[1, 0]) * d
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False

                # 如果梯度的tanθ > 1 角度在45度到90度之间， abs(tanθ)=abs(dy/dx) = 1/d => d=abs(dx/dy)
                elif g_t[i, j] >= 1:
                    d = abs(1 / g_t[i, j])
                    first = eight_neiborhood[0, 1] + (eight_neiborhood[0, 2] - eight_neiborhood[0, 1]) * d
                    second = eight_neiborhood[2, 1] + (eight_neiborhood[2, 0] - eight_neiborhood[2, 1]) * d
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False

                # 如果梯度的tanθ < -1 角度在90度到145度之间， abs(tanθ)=abs(dy/dx) = 1/d => d=abs(dx/dy)
                elif g_t[i, j] <= -1:
                    d = abs(1 / g_t[i, j])
                    first = eight_neiborhood[0, 1] + (eight_neiborhood[0, 0] - eight_neiborhood[0, 1]) * d
                    second = eight_neiborhood[2, 1] + (eight_neiborhood[2, 2] - eight_neiborhood[2, 1]) * d
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False

                # 如果梯度的tanθ∈[-1, 0] 角度在145度到180度之间 abs(tanθ)=abs(dy/dx) = d/1 => d=abs(dy/dx)
                elif -1 < g_t[i, j] < 0:
                    d = abs(g_t[i, j])
                    first = eight_neiborhood[1, 0] + (eight_neiborhood[0, 0] - eight_neiborhood[1, 0]) * d
                    second = eight_neiborhood[1, 2] + (eight_neiborhood[2, 2] - eight_neiborhood[1, 2]) * d
                    if not (g_l[i, j] > first and g_l[i, j] > second):
                        isLocalExtreme = False
            if isLocalExtreme:
                result[i, j] = g_l[i, j]       #非极大值抑制
    # 将大于阈值的点设置为255，也就是梯度大于上阈值的点，直接就可以判定为边界，对处于阈值之间的模糊点进行保留，最后在连接函数来筛选这些模糊点
    result[result >= threshold[1]] = 255
    # 将小于最小阈值的点设置为0，也就是很小的极值当做是局部的最大，但是幅值太低了，直接设置为0，舍弃掉这些
    result[result <= threshold[0]] = 0

    # 进行非极大值抑制后连接所有可能连接的边
    result = judgeConnect(result, threshold)
    result[result != 255] = 0
    return result


def OneDimensionStandardNormalDistribution(x, sigma):
    '''
    计算一维高斯核的工具函数
    :param x:坐标，也就是离中心的距离
    :param sigma: 模糊参数
    :return: 该点的高斯核的参数
    '''
    E = -0.5 / (sigma*sigma)
    return 1/ (math.sqrt(2*math.pi)*sigma)*math.exp(x*x*E)


if __name__ == '__main__':

    # Gaussian_filter_3 = 1.0/16*np.array([(1,2,1),(2,4,2),(1,2,1)]) #Gaussian smoothing kernel when sigma = 0.8, size: 3x3
    # Gaussian_filter_5 = 1.0/159*np.array([
    #     [2,4,5,4,2],
    #     [4,9,12,9,4],
    #     [5,12,15,12,5],
    #     [4,9,12,9,4],
    #     [2,4,5,4,2]
    # ])  #Gaussian smoothing kernel when sigma = 1.4, size: 5x5



    pic_path = 'img'
    pics = os.listdir(pic_path)

    # 读取图片
    for i in pics:
        # 因为png的像素值是0-1所以乘以255，标准化到0-255的范围，和jpg一致

        filename = os.path.join(pic_path, i)
        # [h, w, 3]
        img = plt.imread(filename)
        if i.split('.')[-1] == 'png':
            img = img * 255
        # 在rgb通道上取均值，灰度化
        img = img.mean(axis=-1)

        # 高斯核函数的参数
        sigma = 1.52
        # 卷积核的大小, 11
        dim = int(np.round(6*sigma+1))
        # 将卷积核限定为奇数
        dim = dim + 1 if dim % 2 == 0 else dim
        # 算出一维的高斯核的坐标取值，坐标表示离当前点的距离
        # [5,4,3,2,1,0,1,2,3,4,5]
        linear_Gaussian_filter = [np.abs(t - (dim//2)) for t in range(dim)]

        # 将坐标送入高斯核产生函数中产生高斯核数值
        # [11]
        linear_Gaussian_filter = np.array([OneDimensionStandardNormalDistribution(t, sigma) for t in linear_Gaussian_filter])
        # 高斯核归一化
        linear_Gaussian_filter = linear_Gaussian_filter / linear_Gaussian_filter.sum()

        # 拆分的线性卷积代替二维卷积,进行高斯滤波
        img2 = divided_convolve(linear_Gaussian_filter, img)
        # 也可以直接用二维卷积来做
        # img2 = convolve(Gaussian_filter_5, img, [2, 2, 2, 2], [1, 1])

        # 显示灰度化后的图片
        plt.imshow(img2.astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()

        # 两个sobel矩阵求梯度。sobel算子的正负隐含梯度的正方向为dy向上，dx向左，这里会影响后面的线性插值
        sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        # 对灰度化后的图片求导，也就是用sobel核来卷积,x方向求导
        img_x_grad = convolve(sobel_kernel_x, img2, [1,1,1,1], [1,1])
        plt.imshow(img_x_grad.astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()
        # y方向求导
        img_y_grad = convolve(sobel_kernel_y, img2, [1,1,1,1], [1,1])
        plt.imshow(img_y_grad.astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()
        # 平方开根号得到梯度的幅值
        gradiant_length = (img_x_grad**2 + img_y_grad**2)**(1.0/2)

        # 展示梯度幅值的图像
        img_x_grad = img_x_grad.astype(np.float64)
        img_y_grad = img_y_grad.astype(np.float64)
        # 防止求比值的时候出现Nan
        img_x_grad[img_x_grad == 0] = 0.00000001
        # 梯度的角度
        gradiant_tangent = img_y_grad / img_x_grad

        plt.imshow(gradiant_length.astype(np.uint8), cmap='gray')
        plt.axis('off')
        plt.show()

        #lower_boundary = 50
        # 后处理，非极大值抑制和连接极大值的像素点，形成边界
        final_img = DecideAndConnectEdge(gradiant_length, gradiant_tangent)
        cv2.imshow('edge', final_img.astype(np.uint8))
        cv2.waitKey(0)