import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL import ImageEnhance

# bee3文件夹数据增强
def Enhance_Brightness(image):
    # 变亮，增强因子为0.0将产生黑色图像,为1.0将保持原始图像。
    # 亮度增强
    enh_bri = ImageEnhance.Brightness(image)
    brightness = np.random.uniform(0.6, 1.6)
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def Enhance_Color(image):
    # 色度,增强因子为1.0是原始图像
    # 色度增强
    enh_col = ImageEnhance.Color(image)
    color = np.random.uniform(0.4, 2.6)
    image_colored = enh_col.enhance(color)
    return image_colored


def Enhance_contrasted(image):
    # 对比度，增强因子为1.0是原始图片
    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = np.random.uniform(0.6, 1.6)
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def Add_pepper_salt(image):
    # 增加椒盐噪声
    img = np.array(image)
    rows, cols, _ = img.shape
    random_int = np.random.randint(500, 1000)
    for _ in range(random_int):
        x = np.random.randint(0, rows)
        y = np.random.randint(0, cols)
        if np.random.randint(0, 2):
            img[x, y, :] = 255
        else:
            img[x, y, :] = 0
    img = Image.fromarray(img)
    return img


def process(file_i_path, file_i):
    img = cv2.imread(file_i_path)
    # cv2.imshow('src', img)

    # 1、旋转：可能局部丢失，缺失部分用黑色填充
    rot1 = imutils.rotate(img, angle=45)
    rot2 = imutils.rotate(img, angle=90)
    rot3 = imutils.rotate(img, angle=180)

    # cv2.imshow("Rotated1", rot1)
    # cv2.imshow("Rotated2", rot2)
    # cv2.imshow("Rotated3", rot3)

    cv2.imwrite("./dst_bee/bee3/"+str(file_i)[:-4]+"_1.jpg", rot1)
    cv2.imwrite("./dst_bee/bee3/"+str(file_i)[:-4]+"_2.jpg", rot2)
    cv2.imwrite("./dst_bee/bee3/"+str(file_i)[:-4]+"_3.jpg", rot3)

    # 2、翻转
    imgFlip1 = cv2.flip(img, 0)  # 垂直翻转
    imgFlip2 = cv2.flip(img, 1)  # 水平翻转

    # cv2.imshow("imgFlip1", imgFlip1)
    # cv2.imshow("imgFlip2", imgFlip2)

    cv2.imwrite("./dst_bee/bee3/"+str(file_i)[:-4]+"_4.jpg", rot1)
    cv2.imwrite("./dst_bee/bee3/"+str(file_i)[:-4]+"_5.jpg", rot2)

    # 3、平移
    # 图像向上、右平移
    M1 = np.float32([[1, 0, 0], [0, 1, -20]])
    move1 = cv2.warpAffine(img, M1, (img.shape[1], img.shape[0]))

    M2 = np.float32([[1, 0, 20], [0, 1, 0]])
    move2 = cv2.warpAffine(img, M2, (img.shape[1], img.shape[0]))

    # cv2.imshow("move1", move1)
    # cv2.imshow("move2", move2)

    cv2.imwrite("./dst_bee/bee3/"+str(file_i)[:-4]+"_6.jpg", move1)
    cv2.imwrite("./dst_bee/bee3/"+str(file_i)[:-4]+"_7.jpg", move2)

    img = Image.open(file_i_path)
    img_Brightness = Enhance_Brightness(img)
    img_Color = Enhance_Color(img)
    img_contrasted = Enhance_contrasted(img)
    img_pepper_salt = Add_pepper_salt(img)

    # img_Brightness.show("img_Brightness")
    # img_contrasted.show("img_contrasted")
    # img_pepper_salt.show("img_pepper_salt")

    img_Brightness.save("./dst_bee/bee3/"+str(file_i)[:-4]+"_8.jpg")
    img_Color.save("./dst_bee/bee3/"+str(file_i)[:-4]+"_9.jpg")
    img_contrasted.save("./dst_bee/bee3/"+str(file_i)[:-4]+"_10.jpg")
    img_pepper_salt.save("./dst_bee/bee3/" + str(file_i)[:-4] + "_11.jpg")

    cv2.waitKey()
    cv2.destroyAllWindows()


# 多图/文件夹增强
def AllData(rootpath):
    save_loc = root_path
    i = 0
    for a, b, c in os.walk(root_path):
        for file_i in c:
            file_i_path = os.path.join(a, file_i)
            process(file_i_path, file_i)
            i += 1
            print("picture " + str(i) + ":" + str(file_i) + " is over")
    print("process is over!")

if __name__ == "__main__":
    # TestOneDir()
    # TestOnePic()
    root_path = "./src_bee/bee3"
    AllData(root_path)
