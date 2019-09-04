import numpy as np
import tensorflow as tf


def filter_row(row):
    if 'id' in row:
        return int(row.split(': ')[-1])
    elif 'name' in row:
        return row.split(': ')[-1].replace(
            "'",
            ""
        ).strip("\n")


def load_labels(labels_path):
    # initialize the labels dictionary
    print("[INFO] parsing class labels...")
    
    labels = [
        filter_row(row)
        for row in open(labels_path)
        if 'id' in row or 'name' in row
    ]

    return dict(zip(*[iter(labels)]*2))


class SSD_Model:
    """
    This class makes the heavy assumption that you're using one of the SSD models
    from TensorFlow Object Detection API.
    """

    def __init__(self, model_path='detect.tflite', labels_path='labels_map.pbtxt'):
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.output_details = self.interpreter.get_output_details()
        self.input_details = self.interpreter.get_input_details()
        self.labels = load_labels(labels_path)


    def fprop(self, input_data):
        # check shape
        # (should also check if f32)
        assert input_data.shape == self.input_details[0]['shape'], 'welp'
        
        # set input tensor and forward prop
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # update refs to output tensors
        self.boxes, self.classes, self.scores, self.num_detections = [
            interpreter.get_tensor(output['index']) for output in self.output_details
        ]


    def get_output_tensors(self):
        return self.boxes, self.classes, self.scores, self.num_detections
