import argparse
from model.tf import SSD_TF
from model.tflite import load_labels
from video.streamer import Streamer
import cv2
from os.path import isfile

from colormap import colormap
colors = {i: color for i, color in enumerate(colormap.values())}


RELEASE = 'bwdst_filtered_2000/'


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m",
                "--graph_pb",
                default='./tf_files/' + RELEASE + 'frozen_inference_graph.pb',
                help="path to Tensorflow protobuf object detection model")

ap.add_argument("-c",
                "--graph_config",
                default='./tf_files/' + RELEASE + 'graph.pbtxt',
                help="path to pbtxt graph config file")

ap.add_argument("-l",
                "--label_map",
                default='./tf_files/' + RELEASE + 'label_map.pbtxt',
                help="label map file path")

ap.add_argument("-p",
                "--min_prob",
                type=float,
                default=0.6,
                help="minimum probability to filter detections")

args = vars(ap.parse_args())

# I hereby convene that command line args are prefixed with _ and uppercased
_GRAPH_PB_PATH = args['graph_pb']
_GRAPH_CONFIG_PATH = args['graph_config']
_MIN_CONFIDENCE = args['min_prob']
_LABELMAP_PATH = args['label_map']

if not isfile(_GRAPH_PB_PATH):
    raise
elif not isfile(_GRAPH_CONFIG_PATH):
    raise
elif not isfile(_LABELMAP_PATH):
    raise

print("[INFO] min confidence", _MIN_CONFIDENCE)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting cam stream...")
cam_stream = Streamer()

frame_sample = cam_stream.read()
height, width = list(frame_sample.shape)[:2]

model = SSD_TF(
    frozen_graph_path=_GRAPH_PB_PATH,
    config_pbtxt_path=_GRAPH_CONFIG_PATH
)


def bb_mapper(bb):
    xmin, ymin, xmax, ymax = bb.reshape(4)
    xmin = int(xmin * width); xmax = int(xmax * width)
    ymin = int(ymin * height); ymax = int(ymax * height)
    return (xmin, ymin), (xmax, ymax)


labels = load_labels(_LABELMAP_PATH)


# loop over the frames from the video stream
while True:
    bgr_frame = cam_stream.read()
    
    # model inference
    model.fprop(bgr_frame)
    model.filter_results(_MIN_CONFIDENCE)
    boxes = map(bb_mapper, model.get_bounding_boxes())
    scores = model.get_scores()
    class_idxs = model.get_class_idxs() - 1  # ja ajusta o indice

    # loop over the resulting bounding boxes
    for box, score, class_idx in zip(boxes, scores, class_idxs):
        color = colors[class_idx]
        cv2.rectangle(bgr_frame, box[0], box[1], color, 2)

        # label the rectangle
        label = labels[class_idx]
        vert_offset = 3; y = start_y - vert_offset \
            if start_y - vert_offset > vert_offset \
            else start_y + vert_offset

        text = "{} :: {:.0f}%".format(labels[label_idx], score * 100)
        cv2.putText(
            bgr_frame, text, (box[0][0], y), cv2.FONT_HERSHEY_SIMPLEX,
            .35, 1
        )

    # show the output frame and wait for a key press
    cv2.imshow("Frame", bgr_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
