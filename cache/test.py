# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import json
import math
import re
from PIL import Image
import time

start_time = time.time()


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
    threshold = 30
    # 进行像素级别差异检测
    result = pixel_diff_detection(image1, image2, threshold)

    # 截掉下边任务栏
    # 打开像素差异图像
    image = Image.fromarray(result)
    # 获取原始图像的尺寸
    width, height = image.size
    # 定义裁剪区域的坐标
    x = 0
    y = 0
    w = width
    h = height - 40
    # 截掉下边任务栏后的图像
    cropped_image = image.crop((x, y, x + w, y + h))

    # 确定变化的范围
    image = np.array(cropped_image)
    # 获取图像尺寸
    height, width = image.shape
    # 定义初始值
    top = height
    bottom = 0
    left = width
    right = 0
    # 使用NumPy向量化操作，寻找白色像素
    white_pixels = np.argwhere(image == 255)
    if white_pixels.size > 0:
        top, left = white_pixels[0]
        bottom, right = white_pixels[-1]

    # 计算矩形框的坐标
    x1 = left
    y1 = top
    x2 = right
    y2 = bottom
    # TODO: 检测最匹配的矩形从而获得json数据

    # # 绘制矩形框
    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # # 保存裁剪后的图像
    # cv2.imwrite('difference.png', image)

    # TODO: 根据矩形框在（先or后）原图上去截（一张）图
    # TODO: 修改思路：output两张，分别是在原图（先and后）上去截的图
    # Case 1: Popup several panels
    # Case 2: Close several panels
    # Case 3: Manipulate one panel
    # 读取图像（在谁上截图）
    if case == 1:
        cropped_pre = image2    #lat
    elif case == 2:
        cropped_pre = image1    #origin
    elif case == 3:
        cropped_pre = image1    #origin，其实均可

    # image1 = cv2.imread('./test/origin/screenshot-0.png')
    # image2 = cv2.imread('./test/lat2/screenshot-0.png')

    # 截取图像
    cropped_image = cropped_pre[y1:y2, x1:x2]
    cv2.imwrite('cropped_image.png', cropped_image)


def detect_white_rectangle2(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值处理图像
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # 找到轮廓
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 合并相邻的白色区域
    merged_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 100:  # 忽略面积小于100的区域
            continue
        x, y, w, h = cv2.boundingRect(contour)
        merged_contours.append((x, y, x + w, y + h))

    # 在图像上绘制矩形
    for x1, y1, x2, y2 in merged_contours:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 保存带有红框的图像
    cv2.imwrite('merged_white_regions_marked.jpg', image)

    return merged_contours


# 本来：检测变化最大的矩形（并打红框保存下来）
# 新：检测所有变化的矩形：怎么界定
def detect_white_rectangle(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用闭运算填充矩形内部的空洞
    kernel = np.ones((8, 8), np.uint8)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # 使用膨胀操作填充空洞
    dilated = cv2.dilate(closed, kernel, iterations=1)

    # 找到轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # 在图像上绘制所有轮廓
    # cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    # # 保存带有轮廓的图像
    # cv2.imwrite('all_contours_marked.jpg', image)

    # 清空列表，准备存储最小外接矩形信息
    white_rectangles = []

    # 在图像上绘制矩形或最小外接矩形
    for contour in contours:
        # 绘制矩形
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        white_rectangles.append((x, y, w, h))  # 将矩形信息添加到列表中

        # # 绘制最小外接矩形
        # rect = cv2.minAreaRect(contour)
        # box = cv2.boxPoints(rect)
        # box = np.intp(box)
        # cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
        # rect_points = np.intp(box)  # 最小外接矩形的四个顶点坐标
        # x, y, w, h = cv2.boundingRect(rect_points)  # 获取最小外接矩形的位置和大小
        # white_rectangles.append((x, y, w, h))  # 将最小外接矩形信息添加到列表中

    # 保存带有红框的图像
    cv2.imwrite('difference_marked.jpg', image)

    return white_rectangles  # <class 'list'>


def get_name_after_rectangle(json_file, nums):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        panel_data = json.load(f)

    # 解析JSON数据
    # panel_data = json.loads(data)

    # 遍历panel列表
    for panel in panel_data["panel"]:
        # 获取editing_control列表
        editing_control = panel.get("editing_control", [])

        # 遍历editing_control列表
        for control_list in editing_control:
            # 遍历控件字典
            for control in control_list:
                # 获取矩形坐标
                rectangle = control.get("rectangle", [])

                # 检查是否匹配给定的数字
                if len(rectangle) == 4 and all(num == rect for num, rect in zip(nums, rectangle)):
                    # 获取对应的name后的内容
                    # print('pause')
                    name = control.get("name", "")
                    return name

    for panel in panel_data["menu"]:
        # 获取editing_control列表
        editing_control = panel.get("editing_control", [])

        # 遍历editing_control列表
        for control_list in editing_control:
            # 遍历控件字典
            for control in control_list:
                # 获取矩形坐标
                rectangle = control.get("rectangle", [])

                # 如果name对应的是

                # 检查是否匹配给定的数字
                if len(rectangle) == 4 and all(num == rect for num, rect in zip(nums, rectangle)):
                    # 获取对应的name后的内容
                    # print('pause')
                    name = control.get("name", "")
                    return name
    return None


def calculate_similarity(rect1, rect2):
    x1_diff = abs(rect1[0] - rect2[0])
    y1_diff = abs(rect1[1] - rect2[1])
    x2_diff = abs(rect1[2] - rect2[2])
    y2_diff = abs(rect1[3] - rect2[3])

    similarity = 1 / (1 + math.sqrt(x1_diff ** 2 + y1_diff ** 2 + x2_diff ** 2 + y2_diff ** 2))

    return similarity


def extract_rectangles1(text):
    pattern = r"'rectangle': \[(\d+), (\d+), (\d+), (\d+)\]"
    rectangles = re.findall(pattern, str(text))
    results = []
    for rectangle in rectangles:
        x1, y1, x2, y2 = map(int, rectangle)
        results.append((x1, y1, x2, y2))
        # print(results)

    return results


#     find_best_match(json_file, white_rectangles)

def find_best_match(json_file):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    # print(type(data))
    # print(data)

    # 匹配x1, y1, x2, y2测试示例
    text = data
    rectangles = extract_rectangles1(text)
    # print(rectangles)

    # 计算每个矩形与目标矩形之间的相似度
    best_rectangle = None
    best_similarity = -1
    target = (59, 94, 83, 108)

    for rectangle in rectangles:
        similarity = calculate_similarity(target, rectangle)

        if similarity > best_similarity:
            best_similarity = similarity
            best_rectangle = rectangle

    # # 测试示例
    # # text = "{'panel': [{'name': 'Timeline', 'rectangle': [0, 87, 783, 631]
    # rectangles = extract_rectangles2(text, best_rectangle)
    # # print(rectangles)

    best_rectangle = list(best_rectangle)
    # best_rectangle = [334, 101, 360, 110]
    print("最匹配的矩形的rectangle数据：", best_rectangle)
    # rectangle_list = list(best_rectangle)
    # print("rectangle_list:", rectangle_list)
    # previous_string = find_previous_string(text,rectangle_list)
    # previous_string = find_previous_string(list(best_rectangle))

    previous_string = get_name_after_rectangle(json_file, best_rectangle)

    # print("zuizhong:",previous_string)
    json_str = '''{"name": "''', previous_string, '''", "rectangle":''', best_rectangle, '''}'''
    # print(json_str)
    return previous_string, json_str


def find_and_print(json_data, target_value):
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            if value == target_value:
                # print(json_data)
                return
            else:
                find_and_print(value, target_value)
    elif isinstance(json_data, list):
        for item in json_data:
            if item == target_value:
                # print(json_data)
                return
            else:
                find_and_print(item, target_value)


def find_best_matches(json_file, targets):
    # 读取JSON文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    # print(type(data))
    # print(data)

    # 匹配x1, y1, x2, y2测试示例
    text = data
    rectangles = extract_rectangles1(text)
    # print(rectangles)

    # 计算每个矩形与目标矩形之间的相似度
    best_rectangle = []
    temp = []
    best_similarity = -1
    # target = (52, 23, 393, 669)
    for target in targets:
        for rectangle in rectangles:
            similarity = calculate_similarity(target, rectangle)

            if similarity > best_similarity:
                best_similarity = similarity
                # print(type(rectangle))
                temp = rectangle
        # print(temp)
        best_rectangle.append(temp)

    # best_rectangle = list(best_rectangle)
    # best_rectangle = [5, 45, 389, 67]
    print("最匹配的矩形的rectangle数据：", best_rectangle)
    previous_string = find_and_print(data, best_rectangle)

    # previous_string = get_name_after_rectangle(json_file, best_rectangle)

    # return previous_string


if __name__ == '__main__':
    # 读取两张截图
    # input
    # image1 = cv2.imread('./tmp1/pre.png')
    # image2 = cv2.imread('./tmp1/lat.png')

    image1 = cv2.imread('./test/origin/screenshot-0.png')
    image2 = cv2.imread('./test/lat2/screenshot-0.png')

    # TODO: 加入case判断函数
    case = 2

    # 输出差异截图
    picture_differrence(case, image1, image2)# 结果写入了'cropped_image.png'

    image_path = 'difference.png'

    white_rectangles = detect_white_rectangle(image_path)  # <class 'list'>
    print("白色矩形即有变化的地方的rectangle数据：", white_rectangles)

    # 测试示例
    # input
    # json_file = './tmp1/lat.json'
    json_file = './test/lat2/lat2.json'

    # output
    # find_best_matches(json_file, white_rectangles)
    find_best_match(json_file)
    # json_file = './test/origin/origin.json'
    # find_best_match(json_file, white_rectangles)
    # print(json_str)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("代码运行时间：{:.2f}秒".format(elapsed_time))

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
