import matplotlib.pyplot as plt
import numpy as np

if __name__=='__main__':
    img = plt.imread('./img/raccoon.jpg')
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

    # 灰度化
    img = img.mean(axis=-1)
    #plt.imsave('gray_raccoon.jpg', np.dstack((img.astype(np.uint8), img.astype(np.uint8), img.astype(np.uint8))))
    # 傅里叶变换，变化后的图像的每个点都是复数
    img = np.fft.fft2(img)
    # print(img[55,55])
    # 中心化，此时中间是地坪部分，周围是高频部分，图像大多数是低频信号，所以中间会比较亮，周围会比较暗
    img = np.fft.fftshift(img)
    # 经过傅里叶变化的图像每个像素点都是复数，含有实部和虚部，这里只显示出幅值
    fourier = np.abs(img)
    # 模值有些很大，用log函数抑制一下，才能更好显示在图像中
    magnitude_spectrum = np.log(fourier)

    plt.imshow(magnitude_spectrum.astype(np.uint8), cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()  # image after fourier transform
    #plt.imsave('fourier_raccoon.jpg', 14*np.dstack((magnitude_spectrum.astype(np.uint8),magnitude_spectrum.astype(np.uint8),magnitude_spectrum.astype(np.uint8))))

    x, y = img.shape
    lowF = np.zeros((x, y))
    # 新建一个空白的图片，将里面的每个数值设置为复数，因为傅里叶变换后的图像每个坐标是复数
    lowF = lowF.astype(np.complex128)
    window_shape = (20, 20)
    # 只保留中间的值，其他的都是预先设定的0复数
    lowF[int(x / 2) - window_shape[0]:int(x / 2) + window_shape[0],int(y / 2) - window_shape[1]:int(y / 2) + window_shape[1]] = \
        img[int(x / 2) - window_shape[0]:int(x / 2) + window_shape[0],int(y / 2) - window_shape[1]:int(y / 2) + window_shape[1]]
    # 将低频部分进行傅里叶反变换变回来时域，理论上变换回来的时候是没有虚部，由于计算有误差，因此会带着一个很小的虚部
    lowF_im = np.fft.ifft2(lowF)
    lowF_im = np.abs(lowF_im)
    lowF_im[lowF_im > 255] = 255
    plt.imshow(lowF_im.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()
    #plt.imsave('LowF_raccoon.jpg', np.dstack((lowF_im.astype(np.uint8), lowF_im.astype(np.uint8), lowF_im.astype(np.uint8))))

    highF = np.zeros((x, y))
    highF = highF.astype(np.complex128)
    window_shape = (370, 370)
    highF[0:window_shape[0], :] = img[0:window_shape[0], :]
    highF[x - window_shape[0]:x, :] = img[x - window_shape[0]:x, :]
    highF[:, 0:window_shape[1]] = img[:, 0:window_shape[1]]
    highF[:, y - window_shape[1]:y] = img[:, y - window_shape[1]:y]
    highF_im = np.fft.ifft2(highF)
    highF_im = np.abs(highF_im)
    highF_im[highF_im > 255] = 255
    plt.imshow(highF_im.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()
    #plt.imsave('HighF_raccoon.jpg', np.dstack((highF_im.astype(np.uint8), highF_im.astype(np.uint8), highF_im.astype(np.uint8))))
