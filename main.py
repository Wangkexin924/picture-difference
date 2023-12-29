import cv2
import numpy as np
from PIL import Image
import re
from cnocr import CnOcr
import matplotlib
import json
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time


def pixel_diff_detection(image1, image2, threshold):
    # 将图像转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # print(gray1.shape, gray2.shape)

    # 计算每个像素的差异值
    diff = cv2.absdiff(gray1, gray2)

    # 根据设定的阈值进行二值化处理
    _, binary_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    return binary_diff


def find_string_in_file(file_path, target_string, specified_lines):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                if target_string in lines[i]:
                    start_line = max(0, i - specified_lines)  # 确保起始行号不会小于0
                    return ''.join(lines[start_line:i])  # 返回起始行到目标字符串所在行之前的字符
            return "String not found in the file."
    except FileNotFoundError:
        return "File not found."


def update_rectangle(out, json1, image1, image2, x1, y1, x2, y2):
    for item in out:
        text = item['text']
    # 图里有数字吗
    pattern = r'\d'
    match = re.search(pattern, text)
    if match:
        flag = True
    else:
        flag = False
    if flag:
        # 剔除text中的字母
        output_str = re.sub('[a-zA-Z]', '', text)

        file_path = json1  # 文件路径
        target_string = output_str  # 目标字符串
        specified_lines = 7  # 指定行数
        result = find_string_in_file(file_path, target_string, specified_lines)
        # 从result中提取数字
        numbers = re.findall(r'\d+', result)
        # 将提取的字符串数字转换为整数
        x3, y3, x4, y4 = map(int, numbers)
        # 取最大最小x、y，获得新的矩形
        x1 = min(x1, x2, x3, x4)
        x2 = max(x1, x2, x3, x4)
        y1 = min(y1, y2, y3, y4)
        y2 = max(y1, y2, y3, y4)
        y1, y2, x1, x2 = int(y1), int(y2), int(x1), int(x2)
        # 截取图像
        cropped_pre = image1[y1:y2, x1:x2]
        cropped_lat = image2[y1:y2, x1:x2]
        cv2.imwrite('cropped_pre.png', cropped_pre)
        cv2.imwrite('cropped_lat.png', cropped_lat)


def calculate_extended_rectangle(left, top, right, bottom, width, height, threshold):
    x1 = max(left - threshold, 0)
    y1 = max(top - threshold, 0)
    x2 = min(right + threshold, width)
    y2 = min(bottom + threshold, height - 40)
    return x1, y1, x2, y2


def find_green_boxes(data, rectangles):
    for item in data:
        if isinstance(item, dict):
            properties = item.get("properties")
            if properties is not None:
                rectangle = properties.get("rectangle")
                if rectangle is not None:
                    rectangles.append(rectangle)

        elif isinstance(item, list):
            find_green_boxes(item, rectangles)

    rectangles=[(x1, y1, x2, y2) for x1, y1, x2, y2 in rectangles]
    return rectangles


def calculate_area_percentage(red_box, green_boxes):
    red_area = (red_box[2] - red_box[0]) * (red_box[3] - red_box[1])  # 计算红框面积
    intersected_green_boxes = []  # 存储有相交面积的绿框位置
    areas_percentage = []  # 存储相交面积占绿框面积的百分比

    for green_box in green_boxes:
        green_area = (green_box[2] - green_box[0]) * (green_box[3] - green_box[1])  # 计算绿框面积
        intersection_x1 = max(red_box[0], green_box[0])
        intersection_y1 = max(red_box[1], green_box[1])
        intersection_x2 = min(red_box[2], green_box[2])
        intersection_y2 = min(red_box[3], green_box[3])

        if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
            intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
            intersected_green_boxes.append(green_box)
            percentage = intersection_area / green_area * 100
            areas_percentage.append(percentage)

    return intersected_green_boxes, areas_percentage


def is_one_more(img, red_box, green_boxes):
    # 在图像img上绘制红框（变化区域）和绿框（大panel）

    # # 在图像上绘制红框和绿框
    # cv2.rectangle(img, (12, 98), (1072, 626), (0, 0, 255), 2)  # 绘制红框
    # for green_box in green_boxes:
    #     x1, y1, x2, y2 = green_box
    #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿框

    # 计算红框被绿框分割后的面积和百分比
    green_areas, areas_percentage = calculate_area_percentage(red_box, green_boxes)
    # print(green_areas, areas_percentage)

    # -----从这里，上边获得了percentage，加一个判断

    selected_green_areas, selected_areas_percentage, flag = [], [], 0

    for i in range(len(areas_percentage)):
        if 10 <= areas_percentage[i] <= 60:
            selected_green_areas.append(green_areas[i])
            selected_areas_percentage.append(areas_percentage[i])

    # print("保留的green_areas：", selected_green_areas)
    # print("对应位置的保留的areas_percentage：", selected_areas_percentage)

    if selected_green_areas:  # 不为空
        flag = 1

    return selected_green_areas, selected_areas_percentage, flag


def crop_one_more(red_box, selected_green_areas, image1, image2):
    # 初始化最小外接矩形的坐标
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    # 要把原来的截图框加进来
    selected_green_areas.append(red_box)

    # 遍历所有矩形数据，更新最小外接矩形的坐标
    for rectangle in selected_green_areas:
        x1, y1, x2, y2 = rectangle
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    # 定义最大的矩形
    max_rectangle = (min_x, min_y, max_x, max_y)

    # print("最大的矩形坐标：", max_rectangle)

    # 截取图像
    cropped_pre = image1[min_y:max_y, min_x:max_x]
    cropped_lat = image2[min_y:max_y, min_x:max_x]

    cv2.imwrite('cropped_pre.png', cropped_pre)
    cv2.imwrite('cropped_lat.png', cropped_lat)


def picture_differrence(image1, image2, json1, json2):
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

    with open(json1, "r") as f:
        json1 = json.load(f)
    with open(json2, "r") as f:
        json2 = json.load(f)
    # 检测像素差异
    # 设置阈值（根据实际情况调整）
    threshold = 90
    # 进行像素级别差异检测
    result = pixel_diff_detection(image1, image2, threshold)
    # # 显示二进制差异结果
    # plt.imshow(result, cmap='gray')
    # plt.axis('off')
    # plt.show()

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

    # 找到所有白色像素的位置
    white_pixels = np.where(image == 255)

    # 更新矩形框的边界
    top = white_pixels[0].min()
    bottom = white_pixels[0].max()
    left = white_pixels[1].min()
    right = white_pixels[1].max()

    # 计算矩形框的坐标
    x1 = left
    y1 = top
    x2 = right
    y2 = bottom

    # # 扩大边缘
    # x1, y1, x2, y2 = calculate_extended_rectangle(left, top, right, bottom, width, height, threshold)

    # print(width, height)  # 1920 1080
    # print((x1, y1), (x2, y2))  # (12, 98) (1072, 626)
    # print((x2-x1)*(y2-y1))

    # 根据矩形框在（先or后）原图上去截（一张）图
    # Case 1: Popup several panels
    # Case 2: Close several panels
    # Case 3: Manipulate one panel
    # 读取图像（在谁上截图）
    # if case == 1:
    #     cropped_pre = image2  # lat
    # elif case == 2:
    #     cropped_pre = image1  # origin
    # elif case == 3:
    #     cropped_pre = image1  # origin，其实均可


    # 截取图像
    cropped_pre = image1[y1:y2, x1:x2]
    cropped_lat = image2[y1:y2, x1:x2]

    cv2.imwrite('cropped_pre.png', cropped_pre)
    cv2.imwrite('cropped_lat.png', cropped_lat)

    # 判断截图是否有问题，是否要再多一步
    # 面积比例是否都不在10~60之间，如果有在这个范围的就断定为需要再截一遍图
    red_box = (x1, y1, x2, y2)
    rectangles = []
    green_boxes = find_green_boxes(json2.values(),rectangles)  # json2 lat
    selected_green_areas, selected_areas_percentage, flag = is_one_more(image1, red_box, green_boxes)

    # 需要是否需要截一遍图
    if flag == 1:
        crop_one_more(red_box, selected_green_areas, image1, image2)

    # # 思路：截图到数字、小数后，从json中把前一个panel的rectangle获取然后拼接（前一个panel），给一个更完整的图
    # # 参数可以调整，不够一个panel的情况
    # img_fp = cropped_pre  # NOTE：只借用了pre的信息
    # ocr = CnOcr()  # 所有参数都使用默认值
    # out = ocr.ocr(img_fp)
    #
    # # ocr结果条数较短，可能出现不好辨识的情况
    # if len(out) < 5:
    #     update_rectangle(out, json1, image1, image2, x1, y1, x2, y2)
    #
    # # else: 不行动


if __name__ == '__main__':
    # 记录开始时间
    start_time = time.perf_counter()

    # input
    case: int = 5
    # 暂时不再区分case，只用作简易地更改下边input文件

    image1 = './input/case{}/pre.png'.format(case)
    image2 = './input/case{}/lat.png'.format(case)

    json1 = './input/case{}/pre.json'.format(case)
    json2 = './input/case{}/lat.json'.format(case)

    # 输出差异截图
    # picture_differrence(case, image1, image2)  # 结果输出到'cropped_image.png'
    picture_differrence(image1, image2, json1, json2)  # 结果输出到'cropped_pre.png'和'cropped_lat.png'

    # 记录结束时间
    end_time = time.perf_counter()
    
    # 计算整个代码的运行时间
    run_time = end_time - start_time
    print("代码运行时间：", run_time, "秒")
