import cv2
import numpy as np

class SSD_TF:

    # https://jeanvitor.com/tensorflow-object-detection-opencv/
    # graph.pbtxt has to be generated by running opencv dnn scripts
    # python tf_text_graph_ssd.py --input ./3/frozen_inference_graph.pb \
    # --config ./3/pipeline.config --output ./3/graph.pbtxt

    # https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/


    def __init__(self,
                 frozen_graph_path='./tf_files/3/frozen_inference_graph.pb',
                 config_pbtxt_path='./tf_files/3/graph.pbtxt',
                 labelmap_path=None):
        """
        Args:
            frozen_graph_path (str): Path to frozen graph file.
            config_pbtxt_path (str): Path to config file generated from frozen graph.
        """
        self.fg = frozen_graph_path
        self.gtxt = config_pbtxt_path
        self.net = cv2.dnn.readNetFromTensorflow(frozen_graph_path,
                                                 config_pbtxt_path)


    def fprop(self, inpt_img):
        self.net.setInput(cv2.dnn.blobFromImage(inpt_img, 
            size=(300, 300), 
            swapRB=True,
            crop=False
        ))  # also converts to the proper tensor
        
        # shape could pertain to (batch size, "channels", n_dtections, features)
        self.output_tensor = self.net.forward() # (1, 1, 100, 7)
        # each line in this tensor seems to follow
        # (??, class_idx, confidence, *(coordinates))

        def class_idx_mapper(farray):
            return farray[1]

        def score_mapper(farray):
            return farray[2]

        def bb_mapper(farray):
            # (xmin, ymin, xmax, ymax)
            return farray[3], farray[4], farray[5], farray[6]

        
        self.out_scores = np.array(
            list(map(score_mapper, self.output_tensor[0,0]))
        )
        self.bouding_boxes = np.array(
            list(map(bb_mapper, self.output_tensor[0,0]))
        ).reshape((-1, 4))

        self.class_idxs = np.array(
            list(map(class_idx_mapper, self.output_tensor[0,0]))
        ).astype(int)
        return self


    def filter_results(self, min_prob=.8):
        self.idx = np.argwhere(self.out_scores > min_prob)

    def get_bounding_boxes(self):
        return self.bouding_boxes[self.idx]

    def get_scores(self):
        return self.out_scores[self.idx]

    def get_class_idxs(self):
        return self.class_idxs[self.idx]


if __name__ == '__main__':

    from os.path import isfile

    img_filepath = './samples/sample_b.jpg'

    if not isfile(img_filepath):
        print('image not found ::', img_filepath)
        exit(-2)


    colormap = {
        'alice_blue': (240,248,255),
        'alien_armpit': (132,222,2),
        'amber': (255,126,0),
        'amarath_pink': (241,156,187),
        'fuchsia': (255,0,255)
    }
    colors = {i: color for i, color in enumerate(colormap.values())}


    img = cv2.imread(filename=img_filepath)
    

    height, width = list(img.shape)[:2]

    model = SSD_TF()
    model.fprop(img)

    idx = np.argwhere(model.out_scores > .8)  # min thresh
    bbs = model.bouding_boxes[idx]
    scores = model.out_scores[idx]
    class_idxs = model.class_idxs[idx]#.tolist()

    for bb, score, class_idx in zip(bbs, scores, class_idxs):
        xmin, ymin, xmax, ymax = bb.reshape(4)

        xmin = int(xmin * width)
        xmax = int(xmax * width)
        
        ymin = int(ymin * height)
        ymax = int(ymax * height)

        score = score[0]
        class_idx = class_idx[0]

        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            colors[class_idx - 1],
            #thickness=2
        )

        # label the rectangle
        vert_offset = 3
        y = ymin - vert_offset \
            if ymin - vert_offset > vert_offset \
            else ymin + vert_offset
        #text = "{} :: {:.2f}%".format(label, score * 100)
        text = "class [{}] :: {:.0f}%".format(class_idx, score * 100)
        #print(start_x, y)

        cv2.putText(
            img,
            text,
            (xmin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            .35,
            colors[class_idx - 1], 1
        )


    cv2.imshow('Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
