import argparse
from model.tflite import SSD_TFLite, load_labels
from video.streamer import Streamer
import cv2

from colormap import colormap
colors = {i: color for i, color in enumerate(colormap.values())}

# RELEASE = 'bwdst_mobilenet_v2_ssd_21500steps/'
#RELEASE = 'bwdst_filtered_2000/'
#RELEASE = 'bwdst_filtered_10000/'
#RELEASE = 'bwdst_quantized_12900/'  # remember to change line 58 of model/tflite.py
# RELEASE = 'bcst_10000/'
# RELEASE = 'bcst/'
RELEASE = '3/'

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m",
                "--model",
                default='./tf_files/' + RELEASE + 'detect.tflite',
                help="path to TensorFlow Lite object detection model")
ap.add_argument("-l",
                "--labels",
                default='./tf_files/' + RELEASE + 'label_map.pbtxt',
                help="path to labels file")
ap.add_argument("-c",
                "--confidence",
                type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-d",
                "--detection_engine",
                type=str,
                default="lite",
                help="pick a selection engine: (default) 'lite' to use \
    tf.lite.Interpreter; 'edge' to use Coral's Edge TPU.")


args = vars(ap.parse_args())
# I hereby convene that command line args are prefixed with _ and uppercased
_MODEL_PATH = args['model']
_LABELMAP_PATH = args['labels']
_MIN_CONFIDENCE = args['confidence']
_DETECTION_ENGINE = args['detection_engine']


labels = load_labels(_LABELMAP_PATH)


print(_MODEL_PATH)

model = SSD_TFLite(model_path=_MODEL_PATH, labels_path=_LABELMAP_PATH)

print("[INFO] min confidence", _MIN_CONFIDENCE)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")



#cap = cv2.VideoCapture('./samples/sample_a.mkv')
#cap = cv2.VideoCapture('./samples/sample_b.mkv')
#cap = cv2.VideoCapture('./samples/sample_c.mp4')
cap = Streamer('./samples/sample_d.mkv')
cap.set_detection_model(
    model,
    view_mode='camera',
    nn_input_mode='zero-pad'
)


# loop over the frames from the video stream
while True:

    print(cap.read())
    #ret = cap.read()
    bgr_frame, rgb_frame = cap.next_frame()

    #if ret:

    boxes, classes, scores, num_detections = model.fprop(
        rgb_frame).get_output_tensors(_MIN_CONFIDENCE)
    boxes = cap.boxes2bbs(boxes)
    results = zip(boxes, classes, scores)

    # loop over the results
    for box, label_idx, score in results:

        start_x, start_y = box[0]
        end_x, end_y = box[1]

        cv2.rectangle(bgr_frame, box[1], box[0], colors[label_idx])

        # label the rectangle
        vert_offset = 3
        y = start_y - vert_offset \
            if start_y - vert_offset > vert_offset \
            else start_y + vert_offset

        text = "{} :: {:.0f}%".format(labels[label_idx], score * 100)

        cv2.putText(
            bgr_frame,
            text,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            .35,
            colors[label_idx], 1
        )


    # show the output frame and wait for a key press
    cv2.imshow("Frame", bgr_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
