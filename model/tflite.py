import numpy as np
import tensorflow as tf

# prefixed
WIDTH = 300
HEIGHT = 300


def filter_row(row):
    if 'id' in row:
        return int(row.split(': ')[-1]) - 1
        # look out! label mismatch may happen from this - 1
    elif 'name' in row:
        return row.split(': ')[-1].replace("'", "").strip("\n")


def load_labels(labels_path):
    # initialize the labels dictionary
    print("[INFO] parsing class labels...")

    labels = [
        filter_row(row) for row in open(labels_path)
        if 'id' in row or 'name' in row
    ]

    return dict(zip(*[iter(labels)] * 2))


class SSD_TFLite:
    """
    This class makes the heavy assumption that you're using one of the SSD models
    from TensorFlow Object Detection API.
    """
    def __init__(self,
                 model_path='detect.tflite',
                 labels_path='labels_map.pbtxt'):
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.output_details = self.interpreter.get_output_details()
        self.input_details = self.interpreter.get_input_details()

        self.labels = load_labels(labels_path)
        self.input_shape = self.input_details[0]['shape'][[1, 2]]
        self.width = self.input_shape[0]
        self.height = self.input_shape[1]

    def fprop(self, input_data):
        # check shape
        # (should also check if f32)
        #assert input_data.shape == self.input_details[0]['shape'], 'welp'

        # set input tensor and forward prop
        print(input_data.shape)
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            #input_data.astype('uint8')
            input_data.astype('float32')
        )
        self.interpreter.invoke()

        # update refs to output tensors
        self.boxes, self.classes, self.scores, self.num_detections = [
            self.interpreter.get_tensor(output['index'])
            for output in self.output_details
        ]

        self.classes = self.classes.reshape([-1]).astype(int)
        # print('CALSSA', self.classes)
        # change int classes to their labels
        
        # self.classes = np.array([
        #     self.labels[key] for key in self.classes.reshape([-1]).astype(int)
        # ])

        # reshape for convenience
        self.boxes = self.boxes.reshape([-1, 4])
        self.scores = self.scores.reshape([-1])
        return self

    def get_output_tensors(self, threshold=None):
        """
        threshold : float
            minimum confidence to consider bounding boxes.
        """
        assert (type(threshold) is float) and (threshold >= 0.0) \
            and (threshold <=1.0)

        if threshold is None:
            return self.boxes, self.classes, self.scores, self.num_detections
        else:
            #idx = np.argwhere(self.scores > 1.1).reshape([-1])  # try out this case
            idx = np.argwhere(self.scores > threshold).reshape([-1])
            # print(idx)
            # print(type(self.classes))
            # print(self.boxes[idx])
            # print(self.classes[idx])
            # print(self.scores[idx])
            return self.boxes[idx], self.classes[idx], self.scores[
                idx], idx.size
            # for debugging purposes
            #return [self.boxes[0]], [self.classes[0]], [self.scores[0]], 1


if __name__ == '__main__':

    model = SSD_TFLite('./tf_files/detect.tflite',
                       './tf_files/label_map.pbtxt')
