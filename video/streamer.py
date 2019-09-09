from time import sleep

import imutils
from imutils.video import VideoStream

from PIL import Image
import cv2
import numpy as np


def make_padding(nrows, ncols):
    return np.zeros((nrows, ncols, 3), dtype=np.uint8)


class Streamer(VideoStream):


    def __init__(self, src=0):
        super(Streamer, self).__init__(src=src)
        super(Streamer, self).start()
        sleep(2)
        shape = self.read().shape
        self.height = shape[0]
        self.width = shape[1]


    def set_detection_viewing_mode(self, model, mode='convert'):
        """
        model
            - An instance of SSD class, representing a neural network model.
        mode
            - {'convert', 'reflect'}, convert back to stream size, or reflect
            the view of the neural network
        """
        self.mode = mode
        assert self.mode == 'convert' or self.mode == 'reflect', \
            "Vieweing `mode` should be either \{'convert', 'reflect'\}." \
            + " It was set to '{}' instead".format(mode)
        self.nn_width = model.width
        self.nn_height = model.height
        self.width_ratio = self.width / self.nn_width
        self.height_ratio = self.height / self.nn_height


    def next_frame(self, nn_dims=(None, None)):
        """
        Returns original frame (BGR) for OpenCV and the RGB equivalent.
        
        # there's stuff to work on according to this note:
        # https://www.tensorflow.org/lite/models/object_detection/overview#location

        TODO
        Note that this is hard coded hack and that rgb need not be the same as bgr,
        provided proper conversions are performed before and after feeding to neural network.

        This also has implications for dealing with bounding boxes.
        For the time being, I'm 
        """
        bgr = self.read()  # numpy.ndarray
        
        # resize along the largest dimension (width on laptop camera)
        # (the smallest one will be necessarily smaller than this value)
        # and maintain aspect ratio (in this case results in 225, 300)
        #print(bgr.shape)
        # zero padding on top and bottom
        #bgr = np.concatenate((make_padding(32, 300), bgr, make_padding(33, 300)))
        #print(bgr.shape)

        #bgr = rgb.copy()
        # prepare a neural network friendly version of the frame
        rgb = bgr.copy()
        rgb = cv2.resize(rgb, (300, 300))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.reshape((1, *rgb.shape))  # make this array 4D

        return bgr, rgb#Image.fromarray(frame)
    

    def boxes2bbs(self, boxes):
        # https://stackoverflow.com/a/48915493
        """
        Converts the model output to something 
        more intuitive to draw with opencv, including
        converting relative values to pixel coordinates.

        Also, reconverts to coordinates of some other frame space
        (e.g. webcam feed frame).
        This works by passing re_dimesions, a (width, height) tuple
        describing the dimensions of the visualized frame on screen.
        """

        # print(
        #     'width_ratio, height_ratio =',
        #     (self.width_ratio, self.height_ratio)
        # )

        def process_box(box):
            # https://www.tensorflow.org/lite/models/object_detection/overview#output
            # print(box)
            top, left, bottom, right = box
            # print('(top, left, bottom, right) =', (top, left, bottom, right))

            # convert to pixel values of the neural net input frame
            x_start = np.max([0, np.floor(left * self.nn_width)])
            y_start = np.max([0, np.floor(top * self.nn_height)])
            
            x_end = np.min([self.nn_width, np.floor(right * self.nn_width)])
            y_end = np.min([self.nn_height, np.floor(bottom * self.nn_height)])

            # print('(x_st, y_st, x_end, y_end) =', (x_start, y_start, x_end, y_end))

            # if reprojecting to original camera feed

            if self.mode == 'convert':
                x_start = x_start / self.width_ratio
                x_end = x_end * self.width_ratio
                
                y_start = y_start / self.height_ratio
                y_end = y_end * self.height_ratio
            
            # print(
            #     '(x_st*, y_st*, x_end*, y_end*) =',
            #     (x_start, y_start, x_end, y_end)
            # )
            assert x_start < x_end and y_start < y_end, 'wrong coordinate conversion'
            return [(int(x_start), int(y_start)), (int(x_end), int(y_end))]

        return [process_box(box) for box in boxes]




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

