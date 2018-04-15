from os import listdir
from os.path import isfile, join
import cv2
import pickle

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

def process_video(folder_index):

    from darkflow.net.build import TFNet
    threshold = 0.25

    options = { "model": "darkflow/cfg/yolo.cfg",
        "load": "darkflow/yolo.weights", "threshold": threshold, "gpu": 1 }
    tfnet = TFNet(options)

    video_data = load_video(folder_index)
    objects_t = []
    for i in xrange(len(video_data)):
        print "Processing " + str(i)
        filtered_objs = tfnet.return_predict(video_data[i])
        objects_t.append(filtered_objs)

    with open("filtered_obj_" + str(folder_index) + "_" + str(int(threshold * 100)) + ".pkl", "w") as f_out:
        pickle.dump(objects_t, f_out)

def generate_video(proposals, folder_index, video_name):
    video_data = load_video(folder_index)
    height, width, _ = video_data[0].shape

    print "Generating video size ", str(width), " x ", str(height)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, fourcc, 10, (width, height))

    for t in xrange(len(proposals)):
        filtered_objs = [s[0] for s in proposals[t] if s[0] is not None]
        frame = wrap_bounding_boxes(video_data[t], filtered_objs)
        for movement in proposals[t]:
            start = movement[0]
            end = movement[1]

            if start is None or end is None:
                continue

            x_s, y_s = (start["topleft"]["x"] + start["bottomright"]["x"]) / 2, (start["topleft"]["y"] + start["bottomright"]["y"]) / 2  
            x_e, y_e = (end["topleft"]["x"] + end["bottomright"]["x"]) / 2, (end["topleft"]["y"] + end["bottomright"]["y"]) / 2  
            cv2.line(frame, (x_s, y_s), (x_e, y_e), (70, 120, 70), 4)

        video.write(frame)

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    process_video(1)
    # process_video(3)
    # process_video(4)
    # process_video(5)
    # process_video(6)
    # process_video(7)
    # process_video(8)



