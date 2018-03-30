from os import listdir
from os.path import isfile, join
import cv2

from darkflow.net.build import TFNet

threshold = 0.5

options = { "model": "darkflow/cfg/yolo.cfg",
	"load": "darkflow/yolo.weights", "threshold": threshold }
tfnet = TFNet(options)


def load_video(folder_index):
    def list_image_files(folder_index):
        mypath = "Crowd_PETS09/S2/L1/Time_12-34/View_00" + str(folder_index)
        onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
        return onlyfiles

    image_files = list_image_files(folder_index)
    video_data = [cv2.imread(f) for f in image_files]
    return video_data


def wrap_bounding_boxes(source_image, filtered_objects):
    """
    Displays the bounding boxes overlayed on the RGB Image for visualization

    Credit to object_detector.py written as part of the vision module for the
	autonomous robots developed as part of the CIS 700 coursework
    """

    # copy image so we can draw on it.
    display_image = source_image.copy()
    source_image_width = source_image.shape[1]
    source_image_height = source_image.shape[0]

    # loop through each box and draw it on the image along with a classification label
    for obj_index, obj_dict in enumerate(filtered_objects):

        # calculate box (left, top) and (right, bottom) coordinates
        box_left = obj_dict['topleft']['x']
        box_top = obj_dict['topleft']['y']
        box_right = obj_dict['bottomright']['x']
        box_bottom = obj_dict['bottomright']['y']

        #draw the rectangle on the image.  This is hopefully around the object
        box_color = (0, 255, 0)  # green box
        box_thickness = 2
        cv2.rectangle(display_image, (box_left, box_top),
        	(box_right, box_bottom), box_color, box_thickness)

        # draw the classification label string just above and to the left of the rectangle
        label_background_color = (70, 120, 70) # greyish green background for text
        label_text_color = (255, 255, 255)   # white text
        cv2.rectangle(display_image, (box_left, box_top-20), (box_right, box_top), label_background_color, -1)
        cv2.putText(display_image, obj_dict['label'] + ' : %.2f' % obj_dict['confidence'],
        	(box_left + 5,box_top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    return display_image


video_data = load_video(1)
filtered_objs = tfnet.return_predict(video_data[50])
displayed_image  = wrap_bounding_boxes(video_data[50], filtered_objs)

cv2.imshow('Hit Key to Exit', displayed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()