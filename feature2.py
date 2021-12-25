import pandas as pd
from feature_engineering import eye_feature,mouth_feature
from sklearn.svm import SVC
eye_data = []
mouth_data = []
def getData():
    # 循环示例，后续读取final_data_list.txt

    fileList = open("final_data_list.txt", "r")
    lines = fileList.readlines()
    fileList.close()
    # 画图用factor
    ear_level0y = []
    ear_level1y = []
    ear_level2y = []
    for i in range(0, len(lines)):
        line = lines[i].strip()
        line = line.replace("\\", "/")
        file = pd.read_csv(line)
        if line[11] == '0':
            f = eye_feature(file, 0)
            ear_level0y.append(0)
            eye_data.append(f)
        elif line[11] == '1':
            f = eye_feature(file, 1)
            ear_level1y.append(1)
            eye_data.append(f)
        else:
            f = eye_feature(file, 2)
            ear_level2y.append(2)
            eye_data.append(f)
        mf=mouth_feature(file)
        mouth_data.append(mf)
        # print(mouth_feature(file))

    ear_y = ear_level0y + ear_level1y + ear_level2y
    merge = pd.DataFrame(data=[eye_data, mouth_data,ear_y], index=['eye', 'mouth','label']).T
    merge = merge.sample(frac=1).reset_index(drop=True)
    return merge

