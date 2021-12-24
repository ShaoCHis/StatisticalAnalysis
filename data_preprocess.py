import numpy as np
import pandas as pd
import os


def read_file(file_name, is_skip_header=True):
    if is_skip_header:
        file = pd.read_csv(file_name, header=None, sep=',')
    else:
        file = pd.read_csv(file_name, sep=',')
    return file


def delete_empty_file(root='origin'):
    levels = os.listdir(root)
    with open('non_empty_file_list.txt', 'w') as f:
        for level in levels:
            level_path = os.path.join(root, level)
            file_dirs = os.listdir(level_path)
            for file_name in file_dirs:
                file_path = os.path.join(level_path, file_name)
                file = read_file(file_path)
                if len(file) > 1:
                    f.write(file_path + '\n')


def delete_lack_frame_file(rate=0.08):
    f = open("non_empty_file_list.txt", "r")
    lines = f.readlines()
    f.close()
    count = 0
    with open("non_lack_frame_file_list.txt", "w") as f:
        for line in lines:
            line = line.strip()
            video_file = read_file(line.strip(), False)
            max_frame = max(video_file["frame"].drop_duplicates().tolist())
            if len(video_file) / max_frame / 68 > 1 - rate:
                f.write(line + '\n')
            else:
                count += 1
    print(count / len(lines))


def interpolate(method="linear"):
    f = open("non_lack_frame_file_list.txt", "r")
    lines = f.readlines()
    f.close()
    count = 0
    with open("interpolate_frame_file_list.txt", "w") as f:
        for line in lines:
            line = line.strip()
            video_file = read_file(line.strip(), False)
            video_file.columns = ["No", "frame", "x", "y"]
            line = line.replace("origin", "data")
            frames = video_file["frame"].drop_duplicates().tolist()
            max_frame = max(frames)
            if len(video_file) == max_frame * 68:
                video_file.to_csv(line, index=False)
            else:
                print(line)
                print(frames)
                j = 0
                for i in range(1, max_frame + 1):
                    if frames[j] != i:
                        df1 = video_file.iloc[:(i-1)*68, :]
                        df2 = video_file.iloc[(i-1)*68:, :]
                        df_add = np.full((68, 4), np.nan)
                        df_add[:, 1] = i
                        df_add[:, 0] = range(68)
                        df_add = pd.DataFrame(df_add, columns=["No", "frame", "x", "y"])
                        video_file = pd.concat((df1, df_add, df2), ignore_index=True)
                    else:
                        j += 1
                for i in range(68):
                    video_file[i:-1:68] = video_file[i:-1:68].interpolate(method=method)
                video_file.to_csv(line, index=False)

    print(count / len(lines))



