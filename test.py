from data_preprocess import *

if __name__ == '__main__':
    delete_empty_file('./origin')
    delete_lack_frame_file()
    interpolate()
    split_or_delete_frames()