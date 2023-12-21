import cv2
import numpy as np
from PIL import Image


def picture_differrence(img1, img2, json1, json2):
    # 检测像素差异
    # 设置阈值（根据实际情况调整）
    threshold = 90
    # 进行像素级别差异检测
    # 将图像转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 计算每个像素的差异值
    diff = cv2.absdiff(gray1, gray2)

    # 根据设定的阈值进行二值化处理
    _, binary_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    result = binary_diff

    # 截掉下边任务栏
    # 打开像素差异图像
    image = Image.fromarray(result)
    # 获取原始图像的尺寸
    width, height = image.size
    # print(width, height)    #1920 1080
    # 定义裁剪区域的坐标
    x = 0
    y = 0
    w = width
    h = height - 40
    # 截掉下边任务栏后的图像
    cropped_image = image.crop((x, y, x + w, y + h))
    # cropped_image.save('./cache/no_taskbar.png')

    # 确定变化的矩形
    image = np.array(cropped_image)

    # 获取图像尺寸
    height, width = image.shape
    # 定义初始值
    top = height
    bottom = 0
    left = width
    right = 0

    # 从上方开始遍历像素，找到第一个白色像素的位置
    for y in range(height):
        for x in range(width):
            if image[y, x] == 255:
                if y < top:
                    top = y
                break  # 提前终止内层循环

    # 从下方开始遍历像素，找到第一个白色像素的位置
    for y in range(height - 1, -1, -1):
        for x in range(width):
            if image[y, x] == 255:
                if y > bottom:
                    bottom = y
                break  # 提前终止内层循环

    # 从左侧开始遍历像素，找到第一个白色像素的位置
    for x in range(width):
        for y in range(height):
            if image[y, x] == 255:
                if x < left:
                    left = x
                break  # 提前终止内层循环

    # 从右侧开始遍历像素，找到第一个白色像素的位置
    for x in range(width - 1, -1, -1):
        for y in range(height):
            if image[y, x] == 255:
                if x > right:
                    right = x
                break  # 提前终止内层循环

    # # 扩大边缘
    # # 计算矩形框的坐标
    if left - threshold > 0:
        x1 = left - threshold
    else:
        x1 = 0

    if top - threshold > 0:
        y1 = top - threshold
    else:
        y1 = 0

    if right + threshold < width:
        x2 = right + threshold
    else:
        x2 = width

    if bottom + threshold < height - 40:
        y2 = bottom + threshold
    else:
        y2 = height - 40

    # DONE : 修改思路：output两张，分别是在原图（先and后）上去截的图
    # 截取图像
    fimg1, fimg2 = img1[y1:y2, x1:x2], img2[y1:y2, x1:x2]
    return fimg1, fimg2

