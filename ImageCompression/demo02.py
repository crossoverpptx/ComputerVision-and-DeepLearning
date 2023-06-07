from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog

"""
目录图片压缩方法2
"""

# 尺寸比例
size_ratio = 0.7
# 压缩质量
quality = 70


# 遍历文件夹压缩
def batch_compress(srcPath, distPath):
    # 遍历文件夹
    for filename in os.listdir(srcPath):
        # 目录验证
        if not os.path.exists(distPath):
            os.makedirs(distPath)

        # 拼接完整的文件或文件夹路径
        srcFile = os.path.join(srcPath, filename)
        distFile = os.path.join(distPath, filename)

        # 如果是文件 就调用压缩
        if os.path.isfile(srcFile):
            if is_image(srcFile):
                # 执行压缩操作
                compression(srcFile, distFile)
            else:
                print(distFile + " 文件不是图片，跳过！")
        # 如果是文件夹 就继续递归
        elif os.path.isdir(srcFile):
            batch_compress(srcFile, distFile)


# 文件是否为图片判断
def is_image(srcFile):
    if (srcFile.lower().endswith(
            ('.bmp', '.dib', '.gif', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
        return True
    else:
        return False


# 压缩图片并保存
def compression(srcFile, distFile):
    try:
        # 读取原图
        srcImg = Image.open(srcFile)
        w, h = srcImg.size
        # 重新设置图片尺寸和选项，Image.ANTIALIAS：平滑抗锯齿
        distImg = srcImg.resize((int(w * size_ratio), int(h * size_ratio)), Image.Resampling.LANCZOS)
        # 保存为新图
        distImg.save(distFile, quality=quality)
        print(distFile + " 压缩成功！")
    except Exception as e:
        print(distFile + " 压缩失败！异常信息：", e)


def chooseDir():
    root = tk.Tk()
    root.withdraw()
    choose_dir_path = filedialog.askdirectory()
    if choose_dir_path is None or len(choose_dir_path) <= 0:
        print('没有任何选择，程序退出！')
        exit()
    print('您选择的目录：', choose_dir_path)
    return choose_dir_path


if __name__ == '__main__':
    print("=================开始执行=================")
    # 指定图片目录以及压缩后的图片目录
    #choose_dir_path = chooseDir()
    choose_dir_path = './origin'
    output_dir_path = './origin_output'
    batch_compress(choose_dir_path, output_dir_path)
    print("=================执行结束=================")
    print("压缩后的图片已经保存到：" + output_dir_path)
