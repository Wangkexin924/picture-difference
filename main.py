import cv2
import numpy as np
from PIL import Image
import re
from cnocr import CnOcr
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def pixel_diff_detection(image1, image2, threshold):
    # 将图像转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    print(gray1.shape, gray2.shape)

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


def is_one_more():
    img = cv2.imread('./input/case3/lat.png')
    # TODO：green_boxes大panel需要一个生成输入，现在做的是从json自己手动总结的
    red_box = (12, 98, 1072, 626)  # 红框坐标，变化区域
    green_boxes = [(0, 87, 783, 631), (783, 87, 1562, 631), (0, 631, 284, 1020), (284, 631, 336, 1020),
                   (336, 631, 1443, 1020), (1443, 631, 1562, 1020), (1562, 87, 1920, 1020), (0, 43, 1920, 87),
                   (0, 1020, 1920, 1040), (0, 23, 1920, 42)]
    # 绿框坐标列表，大panel

    # 在图像上绘制红框和绿框
    cv2.rectangle(img, (12, 98), (1072, 626), (0, 0, 255), 2)  # 绘制红框
    for green_box in green_boxes:
        x1, y1, x2, y2 = green_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿框

    # 计算红框被绿框分割后的面积和百分比
    green_areas, areas_percentage = calculate_area_percentage(red_box, green_boxes)
    print(green_areas, areas_percentage)

    # -----从这里，上边获得了percentage，加一个判断

    selected_green_areas, selected_areas_percentage, flag = [], [], 0

    for i in range(len(areas_percentage)):
        if 10 <= areas_percentage[i] <= 60:
            selected_green_areas.append(green_areas[i])
            selected_areas_percentage.append(areas_percentage[i])

    print("保留的green_areas：", selected_green_areas)
    print("对应位置的保留的areas_percentage：", selected_areas_percentage)

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

    print("最大的矩形坐标：", max_rectangle)

    # 截取图像
    cropped_pre = image1[min_y:max_y, min_x:max_x]
    cropped_lat = image2[min_y:max_y, min_x:max_x]

    cv2.imwrite('cropped_pre.png', cropped_pre)
    cv2.imwrite('cropped_lat.png', cropped_lat)


def picture_differrence(image1, image2, json1, json2):
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

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

    # 计算矩形框的坐标
    x1 = left
    y1 = top
    x2 = right
    y2 = bottom

    # # # 扩大边缘
    # # # 计算矩形框的坐标
    # if left - threshold > 0:
    #     x1 = left - threshold
    # else:
    #     x1 = 0
    #
    # if top - threshold > 0:
    #     y1 = top - threshold
    # else:
    #     y1 = 0
    #
    # if right + threshold < width:
    #     x2 = right + threshold
    # else:
    #     x2 = width
    #
    # if bottom + threshold < height - 40:
    #     y2 = bottom + threshold
    # else:
    #     y2 = height - 40

    # print(width, height)  # 1920 1080
    print((x1, y1), (x2, y2))  # (12, 98) (1072, 626)
    # print((x2-x1)*(y2-y1))

    # TODO: 检测最匹配的矩形，从而获得json数据

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

    # # 截取图像
    # cropped_image = cropped_pre[y1:y2, x1:x2]
    # cv2.imwrite('cropped_image.png', cropped_image)

    # DONE: 修改思路：output两张，分别是在原图（先and后）上去截的图

    # 截取图像
    cropped_pre = image1[y1:y2, x1:x2]
    cropped_lat = image2[y1:y2, x1:x2]

    cv2.imwrite('cropped_pre.png', cropped_pre)
    cv2.imwrite('cropped_lat.png', cropped_lat)

    # # TODO: 出现问题截图不全一个panel，是否需要根据json信息
    # 需要判断什么时候去再截一遍图
    # 面积比例是否都不在10~60之间，如果有在这个范围的就断定为需要再截一遍图
    # selected_green_areas, selected_areas_percentage, flag = [], [], 0

    selected_green_areas, selected_areas_percentage, flag = is_one_more()
    red_box = (x1, y1, x2, y2)
    if flag == 1:
        crop_one_more(red_box, selected_green_areas, image1, image2)
    #

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
    # 读取两张截图
    # input

    case: int = 3
    # 暂时不再区分case，只用作简易地更改下边input文件

    image1 = './input/case{}/pre.png'.format(case)
    image2 = './input/case{}/lat.png'.format(case)

    json1 = './input/case{}/pre.json'.format(case)
    json2 = './input/case{}/lat.json'.format(case)

    # 输出差异截图
    # picture_differrence(case, image1, image2)  # 结果输出到'cropped_image.png'
    picture_differrence(image1, image2, json1, json2)  # 结果输出到'cropped_pre.png'和'cropped_lat.png'
