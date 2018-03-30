from os import listdir
from os.path import isfile, join
import cv2

from darkflow.net.build import TFNet

threshold = 0.5

options = { "model": "darkflow/cfg/yolo.cfg", "load": "darkflow/yolo.weights", "threshold": threshold }

def load_video(folder_index):
    def list_image_files(folder_index):
        mypath = "Crowd_PETS09/S2/L1/Time_12-34/View_00" + str(folder_index)
        onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        return onlyfiles

    image_files = list_image_files(folder_index)
    video_data = [cv2.imread(f) for f in image_files]
    return video_data

load_video(1)

