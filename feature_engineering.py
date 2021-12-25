import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 眼睛关键点的索引
# 左眼
left_top_left_eye = 37
left_top_right_eye = 38
left_bottom_left_eye = 41
left_bottom_right_eye = 40
left_left_eye_point = 36
left_right_eye_point = 39
# 右眼
right_top_left_eye = 43
right_top_right_eye = 44
right_bottom_left_eye = 47
right_bottom_right_eye = 46
right_left_eye_point = 42
right_right_eye_point = 45

# 嘴关键点索引
left_mouth_point = 48
right_mouth_point = 54
top_left_mouth = 50
top_middle_mouth = 51
top_right_mouth = 52
bottom_left_mouth = 58
bottom_middle_mouth = 57
bottom_right_mouth = 56
# top_left_mouth = 61
# top_middle_mouth = 62
# top_right_mouth = 63
# bottom_left_mouth = 67
# bottom_middle_mouth = 66
# bottom_right_mouth = 65

# 视频的帧数
video_length = 50

# p70,可以调整眼睛闭合比例的大小即pxx
p = 0.3

# F为疲劳度阈值，f>F即为疲劳
F = 0.3

# 嘴巴纵横比阈值
openThresh = 0.5

# f值，即perclos(按照一条csv的close_thresh计算)值
ear_self_level0 = []
ear_self_level1 = []
ear_self_level2 = []
# f值，按照mean_ear计算
ear_mean_level0 = []
ear_mean_level1 = []
ear_mean_level2 = []
# ear的差值（一条csv中，earList的max-min）
ear_level0_difference = []
ear_level1_difference = []
ear_level2_difference = []
# level0的close_thresh
mean_ear = 0


# 计算眼部的纵横比
def compute_ear(data, frame):
    # (|(p37-p41)|+|(p38-p40)|)/2*|(p36-p39)|
    # 减少计算次数
    index = frame * 68
    # 左眼计算
    left_eye_top_left = index + left_top_left_eye
    left_eye_bottom_left = index + left_bottom_left_eye
    left_eye_top_right = index + left_top_right_eye
    left_eye_bottom_right = index + left_bottom_right_eye
    left_vertical_left = compute_line_distance(data["x"][left_eye_top_left], data["x"][left_eye_bottom_left],
                                               data["y"][left_eye_top_left], data["y"][left_eye_bottom_left])
    left_vertical_right = compute_line_distance(data["x"][left_eye_top_right], data["x"][left_eye_bottom_right],
                                                data["y"][left_eye_top_right], data["y"][left_eye_bottom_right])
    # 得到纵向长度
    left_vertical_length = left_vertical_right + left_vertical_left
    # 计算横向长度
    left_eye_left_index = index + left_left_eye_point
    left_eye_right_index = index + left_right_eye_point
    left_horizontal_length = compute_line_distance(data["x"][left_eye_left_index], data["x"][left_eye_right_index],
                                                   data["y"][left_eye_left_index], data["y"][left_eye_right_index])
    left_ear = left_vertical_length / (2 * left_horizontal_length)
    # 右眼计算
    right_eye_top_left = index + right_top_left_eye
    right_eye_bottom_left = index + right_bottom_left_eye
    right_eye_top_right = index + right_top_right_eye
    right_eye_bottom_right = index + right_bottom_right_eye
    right_vertical_left = compute_line_distance(data["x"][right_eye_top_left], data["x"][right_eye_bottom_left],
                                                data["y"][right_eye_top_left], data["y"][right_eye_bottom_left])
    right_vertical_right = compute_line_distance(data["x"][right_eye_top_right], data["x"][right_eye_bottom_right],
                                                 data["y"][right_eye_top_right], data["y"][right_eye_bottom_right])
    # 得到纵向长度
    right_vertical_length = right_vertical_right + right_vertical_left
    # 计算横向长度
    right_eye_left_index = index + right_left_eye_point
    right_eye_right_index = index + right_right_eye_point
    right_horizontal_length = compute_line_distance(data["x"][right_eye_left_index], data["x"][right_eye_right_index],
                                                    data["y"][right_eye_left_index], data["y"][right_eye_right_index])
    right_ear = right_vertical_length / (2 * right_horizontal_length)
    ear = (left_ear + right_ear) / 2
    return ear


# 目前仅为二分类，即疲劳，不疲劳
# 可以将f作为特征值输入三分类模型进行训练
def compute_perclos(data, type):
    ear_list = []
    for frame in range(0, video_length):
        ear = compute_ear(data, frame)
        ear_list.append(ear)
    if type == 0:
        # 计算阈值
        close_eye_thresh = min(ear_list) + (max(ear_list) - min(ear_list)) * p
        ear_level0_difference.append(max(ear_list) - min(ear_list))
    elif type == 1:
        close_eye_thresh = min(ear_list) + (max(ear_list) - min(ear_list)) * p
        ear_level1_difference.append(max(ear_list) - min(ear_list))
    else:
        close_eye_thresh = min(ear_list) + (max(ear_list) - min(ear_list)) * p
        ear_level2_difference.append(max(ear_list) - min(ear_list))
    # 统计闭眼次数
    close_count = 0
    for ear in ear_list:
        # if ear < close_eye_thresh:
        if ear < 0.276:
            close_count += 1
    f = close_count / video_length
    if type == 0:
        ear_mean_level0.append(f)
    elif type == 1:
        ear_mean_level1.append(f)
    else:
        ear_mean_level2.append(f)
    close_count = 0
    for ear in ear_list:
        if ear < close_eye_thresh:
            close_count += 1
    f = close_count / video_length
    if type == 0:
        ear_self_level0.append(f)
    elif type == 1:
        ear_self_level1.append(f)
    else:
        ear_self_level2.append(f)
    # return f


# 计算嘴部纵横比
def compute_mar(data, frame):
    # (| (p61-p67) | + | (p62 - p66) |+|(p63-p65)|) / 3 * | (p48 - p54) |
    # 减少计算次数
    index = frame * 68
    # 计算纵向长度
    # 左侧索引
    left_top_index = index + top_left_mouth
    left_bottom_index = index + bottom_left_mouth
    left_vertical_length = compute_line_distance(data["x"][left_top_index], data["x"][left_bottom_index],
                                                 data["y"][left_top_index], data["y"][left_bottom_index])
    # 右侧索引
    right_top_index = index + top_right_mouth
    right_bottom_index = index + bottom_right_mouth
    right_vertical_length = compute_line_distance(data["x"][right_top_index], data["x"][right_bottom_index],
                                                  data["y"][right_top_index], data["y"][right_bottom_index])
    # 中部索引
    middle_top_index = index + top_middle_mouth
    middle_bottom_index = index + bottom_middle_mouth
    middle_vertical_length = compute_line_distance(data["x"][middle_top_index], data["x"][middle_bottom_index],
                                                   data["y"][middle_top_index], data["y"][middle_bottom_index])
    vertical_length = left_vertical_length + right_vertical_length + middle_vertical_length
    # 计算横向长度
    mouth_left_index = index + left_mouth_point
    mouth_right_index = index + right_mouth_point
    horizontal_length = compute_line_distance(data["x"][mouth_left_index], data["x"][mouth_right_index],
                                              data["y"][mouth_left_index], data["y"][mouth_right_index])
    mar = vertical_length / (3 * horizontal_length)
    return mar


# 眼部特征
def eye_feature(data, type):
    # 得到f值
    return compute_perclos(data, type)


# 嘴部特征
def mouth_feature(data):
    # 超过5帧大于阈值即计为哈欠
    # 首先初始化哈欠标志位yawn_flag和哈欠计数器yawn_counter为0，当检测到嘴巴张开(MAR > 0.75)
    # 时，yawn_counter自加1，当连续3次检测到MAR > 0.75
    # 即认为驾驶人正在张嘴，开始计时并将yawn_counter置1。当检测到驾驶人嘴巴闭合时，若张嘴持续时间大于等于1.5秒，则认为打了一次哈欠
    # ，哈欠次数yawns自加1。每60秒统计一次哈欠次数，当达到规定的3次 / min时触发警报。
    yawns = 0
    yawn_counter = 0
    for frame in range(0, video_length):
        mar = compute_mar(data, frame)
        if mar > openThresh:
            yawn_counter += 1
            # 这个十五帧需要进行调参
            if yawn_counter >= 10:
                yawns += 1
                yawn_counter = 0
        else:
            yawn_counter = 0
    return yawns


def compute_line_distance(x1, x2, y1, y2):
    x = x1 - x2
    y = y1 - y2
    distance = math.sqrt(x ** 2 + y ** 2)
    return distance


def compute_feature(data, frame):
    return [compute_ear(data, frame), compute_mar(data, frame)]


def return_feature(filePath):
    data = pd.read_csv(line)
    feature = []
    for i in range(0, video_length):
        feature_list.append(compute_feature(data, i))
    return feature, int(filePath[11])


if __name__ == '__main__':
    # 循环示例，后续读取final_data_list.txt
    fileList = open("final_data_list.txt", "r")
    lines = fileList.readlines()
    fileList.close()
    # 画图用 factor
    # ear_level0y = []
    # ear_level1y = []
    # ear_level2y = []
    for i in range(0, len(lines)):
        line = lines[i].strip()
        line = line.replace("\\", "/")
        file = pd.read_csv(line)
        feature_list = []
        if line[11] == '0':
            for i in range(0, video_length):
                feature_list.append(compute_feature(file, i))
            # eye_feature(file, 0)
            # ear_level0y.append(0)
        elif line[11] == '1':
            for i in range(0, video_length):
                feature_list.append(compute_feature(file, i))
            # eye_feature(file, 1)
            # ear_level1y.append(1)
        else:
            for i in range(0, video_length):
                feature_list.append(compute_feature(file, i))
            # eye_feature(file, 2)
            # ear_level2y.append(2)
        # print(mouth_feature(file))
    # ear_y = ear_level0y + ear_level1y + ear_level2y
    # plt.scatter(ear_y, ear_level0_difference + ear_level1_difference + ear_level2_difference, c=ear_y)
    # plt.title("max-min")
    # plt.show()
    # plt.scatter(ear_y, ear_self_level0 + ear_self_level1 + ear_self_level2, c=ear_y)
    # plt.title("self_thresh")
    # plt.show()
    # plt.scatter(ear_y, ear_mean_level0 + ear_mean_level1 + ear_mean_level2, c=ear_y)
    # plt.title("mean_thresh")
    # plt.show()
    exit(0)
