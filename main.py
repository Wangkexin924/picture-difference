import cv2
import numpy as np
from PIL import Image
import re
from cnocr import CnOcr


def pixel_diff_detection(image1, image2, threshold):
    # 将图像转换为灰度图像
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

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


def picture_differrence(image1, image2, json1, json2):
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)

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

    # # 计算矩形框的坐标
    # x1 = left
    # y1 = top
    # x2 = right
    # y2 = bottom

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

    # DONE: 出现问题截图不全一个panel，是否需要根据json信息
    # 思路：截图到数字、小数后，从json中把前一个panel的rectangle获取然后拼接（前一个panel），给一个更完整的图
    # 参数可以调整，不够一个panel的情况
    img_fp = cropped_pre  # NOTE：只借用了pre的信息
    ocr = CnOcr()  # 所有参数都使用默认值
    out = ocr.ocr(img_fp)

    # ocr结果条数较短，可能出现不好辨识的情况
    if len(out) < 5:
        update_rectangle(out, json1, image1, image2, x1, y1, x2, y2)

    # else: 不行动


if __name__ == '__main__':
    # 读取两张截图
    # input

    case: int = 5
    # 暂时不再区分case，只用作简易地更改下边input文件

    # image1 = cv2.imread('./input/case{}/pre.png'.format(case))
    # image2 = cv2.imread('./input/case{}/lat.png'.format(case))

    image1 = './input/case{}/pre.png'.format(case)
    image2 = './input/case{}/lat.png'.format(case)

    json1 = './input/case{}/pre.json'.format(case)
    json2 = './input/case{}/lat.json'.format(case)

    # 输出差异截图
    # picture_differrence(case, image1, image2)  # 结果输出到'cropped_image.png'
    picture_differrence(image1, image2, json1, json2)  # 结果输出到'cropped_pre.png'和'cropped_lat.png'
