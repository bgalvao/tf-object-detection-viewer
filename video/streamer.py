from time import sleep

import imutils
from imutils.video import VideoStream

from PIL import Image
import cv2
import numpy as np


def make_zero_padding(nrows, ncols):
    return np.zeros((nrows, ncols, 3), dtype=np.uint8)

def zero_pad_vertically(rgb_frame, zero_pad):
    return np.concatenate((rgb_frame, zero_pad), axis=0)

def zero_pad_horizontally(rgb_frame, zero_pad):
    return np.concatenate((rgb_frame, zero_pad), axis=1)


class Streamer(VideoStream):


    def __init__(
            self, src=0, use_pi_camera=False, res=(640, 480),
        ):
        super(Streamer, self).__init__(
            src=src,
            usePiCamera=use_pi_camera,
            resolution=res  # doesn't seem to have an effect
        )
        super(Streamer, self).start()
        #sleep(2)
        shape = self.read().shape
        self.height = shape[0]
        self.width = shape[1]
        self.model_set = False


    def set_detection_model(self, model, view_mode='camera',
            nn_input_mode='zero-pad'
        ):
        
        assert view_mode in set(['camera', 'nn']), \
            "Set view_mode to one of \{'camera', 'nn'\}."
        
        assert nn_input_mode in set(['zero-pad', 'distort']),\
            "Set nn_input_mode to one of \{'zero-pad', 'distort'\}"

        # set modes
        self.view_mode = view_mode
        self.nn_input_mode = nn_input_mode

        # set expected input shape by the model
        self.nn_width = model.width
        self.nn_height = model.height
        self.__assert_feed_dimensions_with_model(self.nn_width, self.nn_height)
        
        # set resizing functions
        if self.nn_input_mode == 'distort':
            self.resize_nn_input = self.__resize_with_distortion

            # aspect ratio conversion factors
            # crucial for boxes2bbs
            self.width_ratio = self.width / self.nn_width
            self.height_ratio = self.height / self.nn_height

        elif self.nn_input_mode == 'zero-pad':
            self.resize_nn_input = self.__resize_with_zero_padding

            # should you choose to keep aspect ratio,
            # you have to zero-pad for tflite (might be different for coral)
            # this assumes that nn input shape is smaller than camera feed's
            # in both its dimensions.
            zero_example = np.zeros(shape=(self.height, self.width))
            r_height, r_width = imutils.resize(
                zero_example,
                width=self.nn_width,
                height=self.nn_height
            ).shape
            del zero_example
            
            if (r_width == self.nn_width) and (r_height == self.nn_height):
                self.zero_padding = None
            elif (r_width == self.nn_width) and (r_height != self.nn_height):
                diff = self.nn_height - r_height
                self.zero_padding = make_zero_padding(diff, self.nn_width)
                self.zero_pad = zero_pad_vertically
            else:
                self.zero_padding = make_zero_padding(self.nn_height, diff)
                self.zero_pad = zero_pad_horizontally
        else:
            raise
        
        if self.view_mode == 'camera' and self.nn_input_mode == 'distort':
            self.process_box = self.__un_distort_box
        elif self.view_mode == 'camera' and self.nn_input_mode == 'zero-pad':
            self.process_box = self.__un_zero_pad_box
        elif self.view_mode == 'nn':
            self.process_box = self.__un_zero_pad_box_nn_frame

        self.model_set = True


    def __assert_feed_dimensions_with_model(self, nn_width, nn_height):
        # note that this might just be for tflite
        assert nn_height < self.height or nn_width < self.width,\
            'Unexpected camera feed dimensions; this case is not yet supported'\
            + '. Please pick stream dimensions equal or greater than those of' \
            + ' the neural network (W,H) = ({}, {}) in the constructor.'.format(
                self.nn_width, self.nn_height
            ) + '\n\n - Camera stream dims (W, H) = ({}, {}).'.format(
                self.width, self.height
            )


    def next_frame(self):
        bgr = self.read()  # numpy.ndarray
        nn_input = self.resize_nn_input(bgr.copy())
        
        if self.view_mode == 'nn':
            bgr = nn_input
        
        nn_input = cv2.cvtColor(nn_input, cv2.COLOR_BGR2RGB)
        nn_input = nn_input.reshape((1, *nn_input.shape))

        return bgr, nn_input


    def __resize_with_distortion(self, frame):
        return cv2.resize(frame, (self.nn_width, self.nn_height))


    def __resize_with_zero_padding(self, frame):
        nn_input = imutils.resize(
            frame,
            width=self.nn_width, height=self.nn_height
        )
        return self.zero_pad(nn_input, self.zero_padding)
    

    def boxes2bbs(self, boxes):
        # https://stackoverflow.com/a/48915493
        return [self.process_box(box) for box in boxes]


    def __un_distort_box(self, box):
        # https://www.tensorflow.org/lite/models/object_detection/overview#output
        # print(box)
        top, left, bottom, right = box

        # convert to pixel values of the neural net input frame
        x_start = np.max([0, np.floor(left * self.nn_width)])
        y_start = np.max([0, np.floor(top * self.nn_height)])
        x_end = np.min([self.nn_width, np.floor(right * self.nn_width)])
        y_end = np.min([self.nn_height, np.floor(bottom * self.nn_height)])

        x_start = x_start / self.width_ratio
        x_end = x_end * self.width_ratio        
        y_start = y_start / self.height_ratio
        y_end = y_end * self.height_ratio

        assert x_start < x_end and y_start < y_end, 'wrong coordinate conversion'
        return [(int(x_start), int(y_start)), (int(x_end), int(y_end))]


    def __un_zero_pad_box_nn_frame(self, box, is_final_output=True):
        y_min, x_min, y_max, x_max = box

        # absolute values
        y_min = y_min * self.nn_height
        y_max = y_max * self.nn_height
        x_min = x_min * self.nn_width
        x_max = x_max * self.nn_width

        if is_final_output:
            return [(int(x_min), int(y_min)), (int(x_max), int(y_max))]
        else:
            return y_min, x_min, y_max, x_max


    def __un_zero_pad_box(self, box):
        y_min_raw, x_min_raw, \
            y_max_raw, x_max_raw = self.__un_zero_pad_box_nn_frame(box, False)
        
        # note, you can still save some compute by looking at self.orientation..
        # new relative values
        y_min = y_min_raw / self.nn_height
        y_max = y_max_raw / self.nn_height
        x_min = x_min_raw / self.nn_width
        x_max = x_max_raw / self.nn_width

        # new absolute values (i.e. in pixels)
        y_min = int(y_min * self.height)
        y_max = int(y_max * self.height)
        x_min = int(x_min * self.width)
        x_max = int(x_max * self.width)

        return [(x_min, y_min), (x_max, y_max)]

        



if __name__ == '__main__':

    # simply stream from the camera

    vs = Streamer()

    print('width:', vs.width, '   height:', vs.height)

    while True:
        bgr_frame, rgb_frame = vs.next_frame()
        cv2.imshow("Frame", bgr_frame)

        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# do a bit of cleanup
#cv2.destroyAllWindows()
#vs.stop()

