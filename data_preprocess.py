import pandas as pd
import os


def read_file(file_name):
    file = pd.read_csv(file_name, header=None, sep=',')
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



