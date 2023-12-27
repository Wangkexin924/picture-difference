import cv2
import numpy as np


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


img = cv2.imread('./input/case3/lat.png')
# TODO：green_boxes大panel需要一个生成输入，现在做的是从json自己手动总结的
red_box = (12, 98, 1072, 626)  # 红框坐标，变化区域
green_boxes = [(0, 87, 783, 631), (783, 87, 1562, 631), (0, 631, 284, 1020), (284, 631, 336, 1020), (336, 631, 1443, 1020), (1443, 631, 1562, 1020), (1562, 87, 1920, 1020), (0, 43, 1920, 87), (0, 1020, 1920, 1040), (0, 23, 1920, 42)]
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

selected_green_areas = []
selected_areas_percentage = []

for i in range(len(areas_percentage)):
    if 10 <= areas_percentage[i] <= 60:
        selected_green_areas.append(green_areas[i])
        selected_areas_percentage.append(areas_percentage[i])

print("保留的green_areas：", selected_green_areas)
print("对应位置的保留的areas_percentage：", selected_areas_percentage)

# if selected_green_areas:
#     return True


# -----从这里，判断出来是否要one more

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
cropped_pre = image1[y1:y2, x1:x2]
cropped_lat = image2[y1:y2, x1:x2]

# # 绘制green_areas在lat图上
# image = cv2.imread('./input/case3/lat.png')
#
# # 绘制红色框
# cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
#
# # 显示图像并等待键盘输入
# cv2.imshow('Image with Red Boxes', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# # 加载图片
# image = cv2.imread('./input/case3/lat.png')
# # 数据信息
# data = [
#     [
#         0,
#         87,
#         783,
#         631
#     ],
#     [
#         783,
#         87,
#         1562,
#         631
#     ],
#     [0, 631, 284, 1020],
#     [284, 631, 336, 1020],
#     [
#         336,
#         631,
#         1443,
#         1020
#     ],
#     [
#         1443,
#         631,
#         1562,
#         1020
#     ],
#     [
#         1562,
#         87,
#         1920,
#         1020
#     ],
#     [
#         0,
#         43,
#         1920,
#         87
#     ],
#     [
#         0,
#         1020,
#         1920,
#         1040
#     ],
#     [
#         0,
#         23,
#         1920,
#         42
#     ],
#
# ]
#
# # 绘制矩形框
# for box in data:
#     x1, y1, x2, y2 = box
#     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
# # 保存图片
# cv2.imwrite('./cache/big_panel.png', image)
