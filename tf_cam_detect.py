import argparse
from model.tf import SSD_TF
from video.streamer import Streamer
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-m",
                "--graph_pb",
                default='./tf_files/3/frozen_inference_graph.pb',
                help="path to Tensorflow protobuf object detection model")

ap.add_argument("-c",
                "--graph_config",
                default='./tf_files/3/graph.pbtxt',
                help="path to pbtxt graph config file")

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

print("[INFO] min confidence", _MIN_CONFIDENCE)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting cam stream...")
video_stream = Streamer()

frame_sample = video_stream.read()
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


i = 0

# loop over the frames from the video stream
while True:
    bgr_frame = video_stream.read()
    
    # model inference
    model.fprop(bgr_frame)
    model.filter_results(_MIN_CONFIDENCE)
    boxes = map(bb_mapper, model.get_bounding_boxes())

    if i % 15 == 0:
        print('\n{} detections'.format(model.get_bounding_boxes().shape[0]))
        for j in model.get_scores().round(2):
            print('-', j)

    # loop over the resulting bounding boxes
    for box in boxes:
        cv2.rectangle(bgr_frame, box[0], box[1], (0, 255, 0), 2)

    # show the output frame and wait for a key press
    cv2.imshow("Frame", bgr_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    i += 1