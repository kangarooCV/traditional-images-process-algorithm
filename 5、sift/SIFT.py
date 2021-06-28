# coding: utf-8


'''
sift分为四个步骤：
    1、构建高斯差分金字塔
    2、关键点定位
    3、基于关键点附近提取描述子向量
    4、匹配
'''
import warnings
warnings.filterwarnings("ignore")  #忽略警告
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image



def cal_conv_out_img_size(img_size, filter_size, padding=None, strides=1):
    # 计算卷积输出图像的大小
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



def downsample(img,step = 2):
    '''
    直接进行下采样
    '''
    return img[::step, ::step]

def GuassianKernel(sigma , dim):
    '''
    :param sigma: Standard deviation
    :param dim: dimension(must be positive and also an odd number)
    :return: return the required Gaussian kernel.
    '''
    temp = [t - (dim//2) for t in range(dim)]
    assistant = []
    for i in range(dim):
        assistant.append(temp)
    assistant = np.array(assistant)
    temp = 2*sigma*sigma
    result = (1.0/(temp*np.pi))*np.exp(-(assistant**2+(assistant.T)**2)/temp)
    return result

def getDoG(img,n,sigma0,S = None,O = None):
    '''
    :param img: the original img.
    :param sigma0: 初始sigma的大小. default 1.52 for complicate reasons.
    :param n: 有效高斯差分金字塔的层数，有效层数=层数-2  因为边缘两个层不能求极值
    :param S: 高斯金字塔的层数
    :param k: the ratio of two adjacent stacks' scale.
    :param O: how many octaves do we have.
    :return: the DoG Pyramid
    '''

    # O = log2(min(M,N)) - 3 ,求高斯金字塔的组数
    O = O or int(np.log2(min(img.shape[0], img.shape[1]))) - 3
    # 由有效高斯差分金字塔求高斯金字塔的层数，高斯金字塔的层数等于有效高斯差分金字塔+2+1， 2代表上下两个边缘，1代表高斯金字塔相邻求差
    S = S or n + 3

    # 控制不同尺度，得到尺度空间
    k = 2 ** (1.0 / n)
    # 求出所有高斯模糊系数，包括所有组和所有层
    # 【5， 6】 5组，每组有6层
    sigma = [[(k**s)*sigma0*(1 << o) for s in range(S)] for o in range(O)]
    # 计算下采样图片，构建图像金字塔（未模糊）
    # 原图片下采样step为[0, 2, 4, 8, 16]
    # [[h,w], [h/2, w/2], [h/4, w/4], [h/8, w/8], [h/16, w/16]]
    samplePyramid = [downsample(img, 1 << o) for o in range(O)]

    # 进行高斯模糊得到高斯金字塔
    GuassianPyramid = []
    # 求高斯金字塔
    # 遍历组数
    for i in range(O):
        GuassianPyramid.append([])
        # 遍历层数
        for j in range(S):
            # 用来生成高斯核的dim
            dim = int(6*sigma[i][j] + 1)
            if dim % 2 == 0:
                dim += 1
            # 用高斯核函数进行高斯模糊得到尺度空间
            GuassianPyramid[-1].append(convolve(GuassianKernel(sigma[i][j], dim), samplePyramid[i], [dim//2, dim//2, dim//2, dim//2], [1,1]))
    # 得到高斯差分金字塔（用相邻的两张图片相减），还不是有效高斯差分金字塔
    DoG = [[GuassianPyramid[o][s+1] - GuassianPyramid[o][s] for s in range(S - 1)] for o in range(O)]
    return DoG, GuassianPyramid


def adjustLocalExtrema(DoG, o, s, x, y, contrastThreshold,edgeThreshold, sigma, n, SIFT_FIXPT_SCALE):
    '''
    得到极值点的位置之后，进行亚像素级别的定位
    para1：DoG： 高斯差分金字塔
    para2:o：极值点处于的组数
    para3：s：极值点处于的层数
    para4， para5： 极值点处于图像中的坐标
    para6：对比阈值，是一个超参数
    para7：sigma：1.6
    para8：n有效高斯差分金字塔层数
    '''
    # 泰勒展开，迭代逼近，迭代次数
    SIFT_MAX_INTERP_STEPS = 5
    SIFT_IMG_BORDER = 5
    point = []

    # 有限差分来求导，计算图像的尺度
    img_scale = 1.0 / (255 * SIFT_FIXPT_SCALE)
    #  差分的2△d
    deriv_scale = img_scale * 0.5
    second_deriv_scale = img_scale
    cross_deriv_scale = img_scale * 0.25

    # 通过高斯差分金字塔的组数和层数，回到高斯金字塔中得到图像
    img = DoG[o][s]
    i = 0
    # 开始迭代逼近
    while i < SIFT_MAX_INTERP_STEPS:
        # 防止越界
        if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= img.shape[0] - SIFT_IMG_BORDER:
            return None, None, None, None

        # 得到高斯金字塔中的上下两张图像
        # img = DoG[o][s]
        prev = DoG[o][s - 1]
        next = DoG[o][s + 1]

        # 开始求导，在高斯金字塔上求更准确的极值点的位置
        # 计算x方向，y方向、z方向上的导数
        dD = [ (img[x,y + 1] - img[x, y - 1]) * deriv_scale,
               (img[x + 1, y] - img[x - 1, y]) * deriv_scale,
               (next[x, y] - prev[x, y]) * deriv_scale ]

        # 求二阶导数 dxx = [f(x+1)-f(x) + f(x)-f(x-1)]/dx^2   :通过泰勒展开来得到的推导式
        v2 = img[x, y] * 2
        dxx = (img[x, y + 1] + img[x, y - 1] - v2) * second_deriv_scale
        dyy = (img[x + 1, y] + img[x - 1, y] - v2) * second_deriv_scale
        dss = (next[x, y] + prev[x, y] - v2) * second_deriv_scale
        dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[x - 1, y - 1]) * cross_deriv_scale
        dxs = (next[x, y + 1] - next[x, y - 1] - prev[x, y + 1] + prev[x, y - 1]) * cross_deriv_scale
        dys = (next[x + 1, y] - next[x - 1, y] - prev[x + 1, y] + prev[x - 1, y]) * cross_deriv_scale

        # 黑塞矩阵（Hessian Matrix）
        H=[ [dxx, dxy, dxs],
            [dxy, dyy, dys],
            [dxs, dys, dss]]
        # np.linalg.pinv：求伪逆
        # 公式中x^ = (H)^-1 * dD, 其中x^代表x-x0，也就是展开点相对于极值点的便宜
        X = np.matmul(np.linalg.pinv(np.array(H)), np.array(dD))

        # 这三个是目前求得的极值点相对于真正极值点的偏移量，如果这三个偏移量足够小，就退出迭代，
        # 否则，将现有极值点加上这个偏移，作为新的极值点，然后在新的极值点泰勒展开，再求极值相对现在的偏移量
        xi = -X[2]
        xr = -X[1]
        xc = -X[0]

        # 如果三个分量都小于0.5的时候，认为已经足够逼近极值点了，就退出迭代
        if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
            break

        y += int(np.round(xc))
        x += int(np.round(xr))
        s += int(np.round(xi))
        i += 1
    # 如果迭代次数已经够了，就直接返回精确的像素的坐标
    if i >= SIFT_MAX_INTERP_STEPS:
        return None,x,y,s
    # 如果迭代次数还没达到，但是三个偏移量已经很接近真正的极值，但是因为这个求得的极值坐标超出了原图的范围了，也舍弃。防止出界
    if s < 1 or s > n or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= \
            img.shape[0] - SIFT_IMG_BORDER:
        return None, None, None, None

    t = (np.array(dD)).dot(np.array([xc, xr, xi]))

    contr = img[x, y] * img_scale + t * 0.5
    # 如果该点对比度不够高，也就是不够亮，也当做一个非极值点，舍弃
    if np.abs(contr) * n < contrastThreshold:
        return None,x,y,s

    # 边缘效应去除，避免求出的点是边缘，最好是多个方向都是极值，而不是单个方向是极值
    # 利用Hessian矩阵的迹和行列式计算主曲率的比值
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    if det <= 0 or tr * tr * edgeThreshold >= (edgeThreshold + 1) * (edgeThreshold + 1) * det:
        return None, x, y, s
    # point将是将金字塔坐标还原回原图的坐标， x和y,s是在金字塔中的实际坐标
    point.append((x + xr) * (1 << o))
    point.append((y + xc) * (1 << o))
    point.append(o + (s << 8) + (int(np.round((xi + 0.5)) * 255) << 16))
    point.append(sigma * np.power(2.0, (s + xi) / n)*(1 << o) * 2)
    # [9.255326614998518, 265.65874144931746, 256, 3.626017047847392]
    # 9, 266, 1
    return point, x, y, s

def GetMainDirection(img, r, c, radius, sigma, BinNum):
    '''
    根据精准定位高斯差分金字塔中的关键点的位置，回到高斯金字塔中，找到和关键点最靠近的图像，统计主方向，主要做了以下几件事：
    1、根据模糊系数sigma，找到最相近的图像
    2、以特征点为半径，以该点所在的高斯图像的尺度的1.5倍为半径来进行高斯滤波
    3、用圆内所有的像素的梯度方向和幅值来统计，选出主方向

    para：
    1、高斯金字塔
    2、r, c 就是在高斯金字塔中的x y
    3、radius： 统计半径大小
    4、sigma：该关键点的高斯模糊系数
    5、将360划分为几个方向来统计
    '''
    expf_scale = -1.0 / (2.0 * sigma * sigma)
    X = []
    Y = []
    W = []
    temphist = []
    for i in range(BinNum):
        temphist.append(0.0)
    # 图像梯度直方图统计的像素范围
    k = 0
    for i in range(-radius, radius+1):
        y = r + i
        # 防止越界
        if y <= 0 or y >= img.shape[0] - 1:
            continue
        for j in range(-radius, radius+1):
            x = c + j
            # 防止越界
            if x <= 0 or x >= img.shape[1] - 1:
                continue
            # 有限差分法求梯度，因为只需要统计方向，所以分母的单位变化量给舍弃了
            dx = (img[y, x + 1] - img[y, x - 1])
            dy = (img[y - 1, x] - img[y + 1, x])
            # x和y的梯度方向
            X.append(dx)
            Y.append(dy)
            W.append((i * i + j * j) * expf_scale)
            k += 1
    # 统计有效的特征点的个数，也就是计算了多少个特征点的向量
    length = k

    W = np.exp(np.array(W))
    Y = np.array(Y)
    X = np.array(X)
    # 由x和y的梯度求出角度，然后将角度转换为角度，现在ori是0-360的数字，也有可能是负数，反正就是一圈的角度
    Ori = np.arctan2(Y,X)*180/np.pi
    # 幅值
    Mag = (X**2+Y**2)**0.5

    # 计算直方图的每个bin
    # 遍历每个特征点周围的统计出来的向量
    for k in range(length):
        # 将360度划分为binNum个区间，也就是binNum个方向，然后看该点落在哪个方向区间内
        bin = int(np.round((BinNum / 360.0) * Ori[k]))
        if bin >= BinNum:
            bin -= BinNum
        if bin < 0:
            bin += BinNum
        temphist[bin] += W[k] * Mag[k]

    # smooth the histogram
    # 高斯平滑
    temp = [temphist[BinNum - 1], temphist[BinNum - 2], temphist[0], temphist[1]]
    temphist.insert(0, temp[0])
    temphist.insert(0, temp[1])
    temphist.insert(len(temphist), temp[2])
    temphist.insert(len(temphist), temp[3])      # padding

    hist = []
    for i in range(BinNum):
        hist.append((temphist[i] + temphist[i+4]) * (1.0 / 16.0) + (temphist[i+1] + temphist[i+3]) * (4.0 / 16.0) + temphist[i+2] * (6.0 / 16.0))

    # 得到主方向
    maxval = max(hist)
    # 返回主方向，有时候也可以放回辅方向
    return maxval, hist


def LocateKeyPoint(DoG, sigma,  GuassianPyramid , n, BinNum = 36, contrastThreshold=0.04, edgeThreshold=10.0):

    # 阈值化：abs(val)=0.5*T/n    T就是contrastThreshold， n是有效高斯差分金字塔的层数
    # 如果极值小于这个值，有可能是噪声，不当做极值点，求出极值要大于这个值才当做真正的极值
    SIFT_ORI_SIG_FCTR = 1.52
    SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR
    SIFT_ORI_PEAK_RATIO = 0.8

    SIFT_INT_DESCR_FCTR = 512.0
    # SIFT_FIXPT_SCALE = 48
    SIFT_FIXPT_SCALE = 1

    KeyPoints = []
    # 得到组数和层数
    O = len(DoG)
    S = len(DoG[0])
    # 遍历每一组
    for o in range(O):
        # 遍历组中的每一层
        for s in range(1, S-1):
            # 计算阈值，极值大于阈值才当做是真正的阈值，否则当做是噪声
            threshold = 0.5*contrastThreshold/(n*255*SIFT_FIXPT_SCALE)
            #取得该层的上下层，三层的尺度空间组成的立体中寻找极值点
            img_prev = DoG[o][s-1]
            img = DoG[o][s]
            img_next = DoG[o][s+1]
            # 在28个邻域内求极值，上层有9个，自己这层有8个，下层有9个
            # 遍历中间层的所有像素点
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    # 得到该点的像素值
                    val = img[i, j]
                    # 取出三层的八邻域
                    eight_neiborhood_prev = img_prev[max(0, i - 1):min(i + 2, img_prev.shape[0]), max(0, j - 1):min(j + 2, img_prev.shape[1])]
                    eight_neiborhood = img[max(0, i - 1):min(i + 2, img.shape[0]), max(0, j - 1):min(j + 2, img.shape[1])]
                    eight_neiborhood_next = img_next[max(0, i - 1):min(i + 2, img_next.shape[0]), max(0, j - 1):min(j + 2, img_next.shape[1])]
                    # 判断阈值，该点是极值且该点绝对值大于阈值才当做是真正的阈值，否则当做是噪声，不处理
                    if np.abs(val) > threshold \
                            and ((val > 0 and (val >= eight_neiborhood_prev).all() and (val >= eight_neiborhood).all() and (val >= eight_neiborhood_next).all())
                            or (val < 0 and (val <= eight_neiborhood_prev).all() and (val <= eight_neiborhood).all() and (val <= eight_neiborhood_next).all())):
                        # 确定该点是有效的极值点之后，进行精确定位,通过泰勒展开，得到亚像素级别的极值点位置
                        # points是特征点的坐标映射回原图的坐标和s和高斯模糊系数sigma，x，y，layer是关键点在高斯差分金字塔中的坐标
                        point, x, y, layer = adjustLocalExtrema(DoG, o, s, i, j, contrastThreshold, edgeThreshold, sigma, n, SIFT_FIXPT_SCALE)
                        if point == None:
                            continue

                        # 确定关键点位置之后确定给关键点赋予主方向
                        # 用高斯差分金字塔得到的准确的极值，在高斯金字塔上找到最相近的图像，来统计主方向
                        # 以该特征点为圆心，特征点所在的，关键点返回的sigma最接近的高斯金字塔图像的尺度的1.5倍为半径的园内所有的像素的梯度方向及其梯度的幅值、
                        # 求主方向依然在高斯金字塔上统计，而不是在原图统计
                        scl_octv = point[-1]*0.5/(1 << o)
                        # 得到主方向
                        omax, hist = GetMainDirection(GuassianPyramid[o][layer], x, y, int(np.round(SIFT_ORI_RADIUS * scl_octv)), SIFT_ORI_SIG_FCTR * scl_octv, BinNum)
                        mag_thr = omax * SIFT_ORI_PEAK_RATIO
                        for k in range(BinNum):
                            if k > 0:
                                l = k - 1
                            else:
                                l = BinNum - 1
                            if k < BinNum - 1:
                                r2 = k + 1
                            else:
                                r2 = 0
                            if hist[k] > hist[l] and hist[k] > hist[r2] and hist[k] >= mag_thr:
                                bin = k + 0.5 * (hist[l]-hist[r2]) /(hist[l] - 2 * hist[k] + hist[r2])
                                if bin < 0:
                                    bin = BinNum + bin
                                else:
                                    if bin >= BinNum:
                                        bin = bin - BinNum
                                temp = point[:]
                                temp.append((360.0/BinNum) * bin)
                                KeyPoints.append(temp)

    # 有一个不知道是啥，第三个存的是层数和组数编码加密的结果，后面计算描述符的时候会解码得到组数和层数
    # [n 个 [x, y, ?, 尺度sigma, 角度方向]]
    return KeyPoints


def calcSIFTDescriptor(img,ptf,ori,scl,d,n,SIFT_DESCR_SCL_FCTR = 3.0,SIFT_DESCR_MAG_THR = 0.2,SIFT_INT_DESCR_FCTR = 512.0,FLT_EPSILON = 1.19209290E-07):
    dst = []
    pt = [int(np.round(ptf[0])), int(np.round(ptf[1]))] # 坐标点取整
    cos_t = np.cos(ori * (np.pi / 180)) # 余弦值
    sin_t = np.sin(ori * (np.pi / 180)) # 正弦值
    bins_per_rad = n / 360.0
    exp_scale = -1.0 / (d * d * 0.5)
    hist_width = SIFT_DESCR_SCL_FCTR * scl
    radius = int(np.round(hist_width * 1.4142135623730951 * (d + 1) * 0.5))
    cos_t /= hist_width
    sin_t /= hist_width

    rows = img.shape[0]
    cols = img.shape[1]


    hist = [0.0]*((d+2)*(d+2)*(n+2))
    X = []
    Y = []
    RBin = []
    CBin = []
    W = []

    k = 0
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            c_rot = j * cos_t - i * sin_t
            r_rot = j * sin_t + i * cos_t
            rbin = r_rot + d // 2 - 0.5
            cbin = c_rot + d // 2 - 0.5
            r = pt[1] + i
            c = pt[0] + j

            if rbin > -1 and rbin < d and cbin > -1 and cbin < d and r > 0 and r < rows - 1 and c > 0 and c < cols - 1:
                dx = (img[r, c+1] - img[r, c-1])
                dy = (img[r-1, c] - img[r+1, c])
                X.append(dx)
                Y.append(dy)
                RBin.append(rbin)
                CBin.append(cbin)
                W.append((c_rot * c_rot + r_rot * r_rot) * exp_scale)
                k+=1

    length = k
    Y = np.array(Y)
    X = np.array(X)
    Ori = np.arctan2(Y,X)*180/np.pi
    Mag = (X ** 2 + Y ** 2) ** 0.5
    W = np.exp(np.array(W))

    for k in range(length):
        rbin = RBin[k]
        cbin = CBin[k]
        obin = (Ori[k] - ori) * bins_per_rad
        mag = Mag[k] * W[k]

        r0 = int(rbin)
        c0 = int(cbin)
        o0 = int(obin)
        rbin -= r0
        cbin -= c0
        obin -= o0

        if o0 < 0:
            o0 += n
        if o0 >= n:
            o0 -= n
        # 线性插值
        # histogram update using tri-linear interpolation
        v_r1 = mag * rbin
        v_r0 = mag - v_r1

        v_rc11 = v_r1 * cbin
        v_rc10 = v_r1 - v_rc11

        v_rc01 = v_r0 * cbin
        v_rc00 = v_r0 - v_rc01

        v_rco111 = v_rc11 * obin
        v_rco110 = v_rc11 - v_rco111

        v_rco101 = v_rc10 * obin
        v_rco100 = v_rc10 - v_rco101

        v_rco011 = v_rc01 * obin
        v_rco010 = v_rc01 - v_rco011

        v_rco001 = v_rc00 * obin
        v_rco000 = v_rc00 - v_rco001

        idx = ((r0 + 1) * (d + 2) + c0 + 1) * (n + 2) + o0
        hist[idx] += v_rco000
        hist[idx+1] += v_rco001
        hist[idx + (n+2)] += v_rco010
        hist[idx + (n+3)] += v_rco011
        hist[idx+(d+2) * (n+2)] += v_rco100
        hist[idx+(d+2) * (n+2)+1] += v_rco101
        hist[idx+(d+3) * (n+2)] += v_rco110
        hist[idx+(d+3) * (n+2)+1] += v_rco111

    # finalize histogram, since the orientation histograms are circular
    for i in range(d):
        for j in range(d):
            idx = ((i+1) * (d+2) + (j+1)) * (n+2)
            hist[idx] += hist[idx+n]
            hist[idx+1] += hist[idx+n+1]
            for k in range(n):
                dst.append(hist[idx+k])

    # copy histogram to the descriptor,
    # apply hysteresis thresholding
    # and scale the result, so that it can be easily converted
    # to byte array
    nrm2 = 0
    length = d * d * n
    for k in range(length):
        nrm2 += dst[k] * dst[k]
    thr = np.sqrt(nrm2) * SIFT_DESCR_MAG_THR

    nrm2 = 0
    for i in range(length):
        val = min(dst[i], thr)
        dst[i] = val
        nrm2 += val * val
    nrm2 = SIFT_INT_DESCR_FCTR / max(np.sqrt(nrm2), FLT_EPSILON)
    for k in range(length):
        dst[k] = min(max(dst[k] * nrm2,0),255)

    return dst


def calcDescriptors(gpyr,keypoints,SIFT_DESCR_WIDTH = 4,SIFT_DESCR_HIST_BINS = 8):
    # SIFT_DESCR_WIDTH = 4，描述直方图的宽度
    # SIFT_DESCR_HIST_BINS = 8
    d = SIFT_DESCR_WIDTH
    n = SIFT_DESCR_HIST_BINS
    descriptors = []

    # 遍历每个关键点
    for i in range(len(keypoints)):
        kpt = keypoints[i]
        o = kpt[2] & 255
        s = (kpt[2] >> 8) & 255  # 该特征点所在的组序号和层序号
        scale = 1.0 / (1 << o)   # 缩放倍数
        size = kpt[3] * scale  # 该特征点所在组的图像尺寸
        ptf = [kpt[1] * scale, kpt[0] * scale]  # 该特征点在金字塔组中的坐标
        img = gpyr[o][s]   # 该点所在的金字塔图像

        descriptors.append(calcSIFTDescriptor(img, ptf, kpt[-1], size * 0.5, d, n))
    return descriptors


def SIFT(img, showDoGimgs = False):
    '''
    提取图像的sift特征描述子
    :param img:图片 [h,w] || [h,w,3]
    :return: KeyPoints关键点位置, discriptors关键点描述子
    '''
    # 设置尺度空间的初始sigma
    SIFT_SIGMA = 1.6
    SIFT_INIT_SIGMA = 0.5  # 假设的摄像头的尺度
    sigma0 = np.sqrt(SIFT_SIGMA**2-SIFT_INIT_SIGMA**2)

    # n是有效差分高斯金字塔的层数，用于求高斯金字塔的层数
    n = 3
    # 计算得到高斯金字塔和高斯差分金字塔,因为做差了，所以层数会少1
    # [5, 5]  [5, 6]
    DoG, GuassianPyramid = getDoG(img, n, sigma0)
    # 展示高斯差分金字塔
    if showDoGimgs:
        for i in DoG:
            for j in i:
                plt.imshow(j.astype(np.uint8), cmap='gray')
                plt.axis('off')
                plt.show()

    # 由高斯金字塔、高斯差分金字塔求关键点定位,中间件涉及泰勒展开迭代算法来求精确定位
    # [n, 5] n个关键点，5代表[x, y, ?, 尺度sigma, 角度方向]， ？记录的是组数和层数编码后的结果，可以解码得到层数和组数
    KeyPoints = LocateKeyPoint(DoG, SIFT_SIGMA, GuassianPyramid, n)

    # 由关键点周围的区域的性质构架描述符向量（128维）
    # 先将关键点附近的图像旋转到主方向上，然后用现行差值得到摆正的图像，再进行特征抽取
    # 类似关键点向量，在关键点附近，划分16小格，统计每个小格里面的梯度的幅值和方向，分为8个方向，也就是16*8=128维度的向量
    discriptors = calcDescriptors(GuassianPyramid,KeyPoints)

    return KeyPoints, discriptors


def Lines(img,info,color = (255,0,0),err = 700):

    if len(img.shape) == 2:
        result = np.dstack((img,img,img))
    else:
        result = img
    k = 0
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            temp = (info[:,1]-info[:,0])
            A = (j - info[:,0])*(info[:,3]-info[:,2])
            B = (i - info[:,2])*(info[:,1]-info[:,0])
            temp[temp == 0] = 1e-9
            t = (j-info[:,0])/temp
            e = np.abs(A-B)
            temp = e < err
            if (temp*(t >= 0)*(t <= 1)).any():
                result[i,j] = color
                k+=1
    print(k)

    return result

def drawLines(X1,X2,Y1,Y2,dis,img,num = 10):

    info = list(np.dstack((X1,X2,Y1,Y2,dis))[0])
    info = sorted(info,key=lambda x:x[-1])
    info = np.array(info)
    info = info[:min(num,info.shape[0]),:]
    img = Lines(img,info)
    #plt.imsave('./5、sift/3.jpg', img)

    if len(img.shape) == 2:
        plt.imshow(img.astype(np.uint8),cmap='gray')
    else:
        plt.imshow(img.astype(np.uint8))
    plt.axis('off')
    #plt.plot([info[:,0], info[:,1]], [info[:,2], info[:,3]], 'c')
    # fig = plt.gcf()
    # fig.set_size_inches(int(img.shape[0]/100.0),int(img.shape[1]/100.0))
    #plt.savefig('./5、sift/2.jpg')
    plt.show()


if __name__ == '__main__':
    origimg = plt.imread('./SIFTimg/3.jpeg')
    img = origimg.mean(axis=-1) if len(origimg.shape)==3 else origimg

    # 计算关键点和描述符
    keyPoints, discriptors = SIFT(img)


    origimg2 = plt.imread('./SIFTimg/4.jpeg')
    img2 = origimg2.mean(axis=-1) if len(origimg.shape) == 3 else origimg2

    # 计算
    ScaleRatio = img.shape[0]*1.0/img2.shape[0]

    img2 = np.array(Image.fromarray(img2).resize((int(round(ScaleRatio * img2.shape[1])),img.shape[0]), Image.BICUBIC))
    # 计算关键点和描述符
    keyPoints2, discriptors2 = SIFT(img2,True)

    # 使用knn算法来匹配描述符
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(discriptors, [0]*len(discriptors))
    match = knn.kneighbors(discriptors2, n_neighbors=1, return_distance=True)

    keyPoints = np.array(keyPoints)[: ,:2]
    keyPoints2 = np.array(keyPoints2)[:,:2]

    keyPoints2[:, 1] = img.shape[1] + keyPoints2[:, 1]

    origimg2 = np.array(Image.fromarray(origimg2).resize((img2.shape[1],img2.shape[0]), Image.BICUBIC))
    result = np.hstack((origimg,origimg2))


    keyPoints = keyPoints[match[1][:,0]]

    X1 = keyPoints[:, 1]
    X2 = keyPoints2[:, 1]
    Y1 = keyPoints[:, 0]
    Y2 = keyPoints2[:, 0]

    # 画线
    drawLines(X1,X2,Y1,Y2,match[0][:,0],result)



# pt_key=AAJg1pqQADDImHMG6kDRguk6AIObsbcvWD3RhUigfuiXMXd7l4bW1H2wT3hitabhrppRhLKZ7Do;
# pt_pin=%E8%B4%AD%E7%89%A9l;