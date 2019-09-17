import argparse
from model.ssd import SSD_Model
from video.streamer import Streamer
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-m", "--model", default='./tf_files/detect.tflite',
    help="path to TensorFlow Lite object detection model"
)
ap.add_argument(
    "-l", "--labels", default='./tf_files/label_map.pbtxt',
    help="path to labels file"
)
ap.add_argument(
    "-c", "--confidence", type=float, default=0.1,
    help="minimum probability to filter weak detections"
)
ap.add_argument(
    "-d", "--detection_engine", type=str, default="lite",
    help="pick a selection engine: (default) 'lite' to use \
    tf.lite.Interpreter; 'edge' to use Coral's Edge TPU."
)
args = vars(ap.parse_args())
# I hereby convene that command line args are prefixed with _ and uppercased
_MODEL_PATH = args['model']
_LABELMAP_PATH = args['labels']
_MIN_CONFIDENCE = args['confidence']
_DETECTION_ENGINE = args['detection_engine']

# note that to use the edge TPU,
# as of now, you really have to use a UINT8 Quantized model.

print(_MODEL_PATH)

model = SSD_Model(
    model_path=_MODEL_PATH,
    labels_path=_LABELMAP_PATH
)

print("[INFO] min confidence", _MIN_CONFIDENCE)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
video_stream = Streamer()
video_stream.set_detection_model(
    model,
    view_mode='camera',
    nn_input_mode='zero-pad'
)




# loop over the frames from the video stream
while True:
    
    # there's stuff to work on according to this note:
    # https://www.tensorflow.org/lite/models/object_detection/overview#location
    bgr_frame, rgb_frame = video_stream.next_frame()
    boxes, classes, scores, num_detections = model.fprop(
        rgb_frame
    ).get_output_tensors(_MIN_CONFIDENCE)
    boxes = video_stream.boxes2bbs(boxes)
    results = zip(boxes, classes, scores)

    # loop over the results https://58surf.com/pt/pt/catalogsearch/result/?q=catch
    for box, label, score in results:
        # extract the bounding box and box and predicted class label
        # box = r.bounding_box.flatten().astype("int")
        # (startX, startY, endX, endY) = box
        # label = labels[r.label_id]
        #print('P1:', box[0], ' ::  P2:', box[1], '\n\n')

        start_x, start_y = box[0]
        end_x, end_y = box[1]

        # draw the bounding box and label on the image
        # if box[0][0] < box[1][0]:
        #     print(box[0][0], box[1][0])
        #     raise
        # #assert box[0][1] < box[1][1]
        cv2.rectangle(bgr_frame, box[1], box[0], (0, 255, 0))
        
        # label the rectangle
        y = start_y - 15 if start_y - 15 > 15 else start_y + 15
        #text = "{} :: {:.2f}%".format(label, score * 100)
        text = "{:.2f}%".format(score * 100)
        #print(start_x, y)
        
        cv2.putText(
            bgr_frame, text, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX,
            .35, (0, 255, 0), 2
        )
        
        # cv2.putText(
        #     bgr_frame, text, (start_x, y),
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     int(1), (0, 255, 0)#, int(2)
        # )

    # show the output frame and wait for a key press
    cv2.imshow("Frame", bgr_frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

