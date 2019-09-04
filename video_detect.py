import argparse
from model.ssd import SSD_Model

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
    "-c", "--confidence", type=float, default=0.3,
    help="minimum probability to filter weak detections"
)
ap.add_argument(
    "-d", "--detection_engine", type=str, default="lite",
    help="pick a selection engine: (default) 'lite' to use \
    tf.lite.Interpreter; 'edge' to use Coral's Edge TPU."
)
args = vars(ap.parse_args())

# note that to use the edge TPU,
# as of now, you really have to use a UINT8 Quantized model.

print(args['model'])

model = SSD_Model(
    model_path=args["model"],
    labels_path=args["labels"]
)


