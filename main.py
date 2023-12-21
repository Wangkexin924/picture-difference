import cv2
import numpy as np
from PIL import Image


def pixel_diff_detection(image1, image2, threshold):
    # 将图像转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算每个像素的差异值
    diff = cv2.absdiff(gray1, gray2)

    # 根据设定的阈值进行二值化处理
    _, binary_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return binary_diff


def picture_differrence(case, image1, image2):
    # 检测像素差异
    # 设置阈值（根据实际情况调整）
    threshold = 90
    # 进行像素级别差异检测
    result = pixel_diff_detection(image1, image2, threshold)

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
    cropped_image.save('./cache/no_taskbar.png')

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

    # # 使用NumPy向量化操作，寻找白色像素
    # white_pixels = np.argwhere(image == 255)
    # if white_pixels.size > 0:
    #     top, left = white_pixels[0]
    #     bottom, right = white_pixels[-1]

    # 计算矩形框的坐标
    x1 = left
    y1 = top
    x2 = right
    y2 = bottom

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

    # print(width, height)  # 1920 1080
    # print((x1, y1), (x2, y2))  # (8, 98) (754, 626)


    # # 读取原图
    # original_image = cv2.imread('no_taskbar.png')
    #
    # # 在原图上画矩形框
    # cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 这里的颜色和线宽可以根据需要进行调整
    #
    # # 保存带有矩形框的图片
    # cv2.imwrite('image_with_rectangle.jpg', original_image)

    # TODO: 检测最匹配的矩形，从而获得json数据

    # 根据矩形框在（先or后）原图上去截（一张）图
    # TODO: 修改思路：output两张，分别是在原图（先and后）上去截的图
    # Case 1: Popup several panels
    # Case 2: Close several panels
    # Case 3: Manipulate one panel
    # 读取图像（在谁上截图）
    if case == 1:
        cropped_pre = image2  # lat
    elif case == 2:
        cropped_pre = image1  # origin
    elif case == 3:
        cropped_pre = image1  # origin，其实均可

    # image1 = cv2.imread('./test/origin/screenshot-0.png')
    # image2 = cv2.imread('./test/lat2/screenshot-0.png')

    # 截取图像
    cropped_image = cropped_pre[y1:y2, x1:x2]
    cv2.imwrite('cropped_image.png', cropped_image)


if __name__ == '__main__':
    # 读取两张截图
    # input

    # TODO: 加入case判断函数
    case: int = 2

    image1 = cv2.imread('./input/case{}/pre.png'.format(case))
    image2 = cv2.imread('./input/case{}/lat.png'.format(case))

    # case = 3

    # 输出差异截图
    picture_differrence(case, image1, image2)  # 结果输出到'cropped_image.png'
