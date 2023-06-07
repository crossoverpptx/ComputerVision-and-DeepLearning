import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 单个图片数据增强

def DotMatrix(A, B):
    '''
    A,B:需要做乘法的两个矩阵，注意输入矩阵的维度ndim是否满足乘法要求（要做判断）
    '''

    return np.matmul(A, B)


class Img:
    def __init__(self, image, rows, cols, center=[0, 0]):
        self.src = image  # 原始图像
        self.rows = rows  # 原始图像的行
        self.cols = cols  # 原始图像的列
        self.center = center  # 旋转中心，默认是[0,0]
        self.rotate = False

    def Move(self, delta_x, delta_y):
        '''
        本函数处理生成做图像平移的矩阵
        '''
        self.transform = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])
        self.rotate = False

    def Zoom(self, factor):  # 缩放
        # factor>1表示缩小；factor<1表示放大
        self.transform = np.array([[factor, 0, 0], [0, factor, 0], [0, 0, factor]])
        self.rotate = False

    def Horizontal(self):
        '''水平镜像
        镜像的这两个函数，因为原始图像读进来后是height×width×3,和我们本身思路width×height×3相反
        所以造成了此处水平镜像和垂直镜像实现的效果是反的'''
        self.transform = np.array([[-1, 0, self.rows], [0, 1, 0], [0, 0, 1]])
        self.rotate = False

    def Vertically(self):
        # 垂直镜像，注意实现原理的矩阵和最后实现效果是和水平镜像是反的
        self.transform = np.array([[1, 0, 0],
                                   [0, -1, self.cols],
                                   [0, 0, 1]])
        self.rotate = False

    def Rotate(self, beta):  # 旋转
        # beta>0表示逆时针旋转；beta<0表示顺时针旋转
        self.transform = np.array([[math.cos(beta), -math.sin(beta), 0],
                                   [math.sin(beta), math.cos(beta), 0],
                                   [0, 0, 1]])
        self.rotate = True

    def Process(self):
        if self.rotate:
            self.center = [int(self.rows / 2), int(self.cols / 2)]
            # 初始化定义目标图像，具有3通道RBG值
        self.dst = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)

        # 提供for循环，遍历图像中的每个像素点，然后使用矩阵乘法，找到变换后的坐标位置
        for i in range(self.rows):
            for j in range(self.cols):

                src_pos = np.array([i - self.center[0], j - self.center[1], 1])  # 设置原始坐标点矩阵
                [x, y, z] = DotMatrix(self.transform, src_pos)  # 和对应变换做矩阵乘法

                x = int(x) + self.center[0]
                y = int(y) + self.center[1]

                if x >= self.rows or y >= self.cols or x < 0 or y < 0:
                    self.dst[i][j] = 255  # 处理未落在原图像中的点的情况
                else:
                    self.dst[i][j] = self.src[x][y]  # 使用变换后的位置


if __name__ == '__main__':
    infer_path = r'test.jpg'  # 要处理的单个图片地址
    imgv = Image.open(infer_path)  # 打开图片
    plt.imshow(imgv)  # 根据数组绘制图像
    plt.show()  # 显示图像

    rows = imgv.size[1]
    cols = imgv.size[0]
    print(rows, cols)  # 注意此处rows和cols的取值方式

    imgv = np.array(imgv)  # 从图像生成数组
    img = Img(imgv, rows, cols, [0, 0])  # 生成一个自定Img类对象[0,0]代表处理的中心点

    img.Vertically()  # 选择处理矩阵
    img.Process()  # 进行矩阵变换
    img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
    plt.imshow(img2)
    print("水平翻转")
    plt.show()

    img.Horizontal()  # 选择处理矩阵
    img.Process()  # 进行矩阵变换
    img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
    plt.imshow(img2)
    print("垂直翻转")
    plt.show()

    img.Rotate(math.radians(180))  # 选择处理矩阵
    img.Process()  # 进行矩阵变换
    img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
    plt.imshow(img2)
    print("旋转")
    plt.show()

    img.Move(-50, -50)  # 选择处理矩阵
    img.Process()  # 进行矩阵变换
    img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
    plt.imshow(img2)
    print("平移")
    plt.show()

    img.Zoom(0.5)  # 选择处理矩阵
    img.Process()  # 进行矩阵变换
    img2 = Image.fromarray(img.dst)  # 从处理后的数组生成图像
    plt.imshow(img2)
    print("放缩")
    plt.show()

    '''
    img.Vertically() #镜像(0,0)
    img.Horizontal() #镜像（0，0）
    img.Rotate(math.radians(180))  #旋转点选择图像大小的中心点
    img.Move(-50,-50) #平移
    img.Zoom(0.5) #缩放
    '''

