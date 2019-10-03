# TensorFlow object detection viewer



This repo includes code to perform, and above all, visualize, object detection on:

- a) a cam feed (your webcam) :: `*_cam_detect.py`

- b) video feed (some video saved on your disk) :: `*_video_detect.py`

and so far supports two detection model formats:

- i) Tensorflow frozen graph (protobuf) :: `tf_*`
- ii) Tensorflow Lite (only with cam feed for now) :: `tflite_*`
- iii) Coral (yet to be supported)

Currently supported implementations are indicated by the `*.py` scripts at the root of this repo.



## how to pass your model files

You need to have these files depending on what you want to run.

| Model format        | Essential files (names are not strict ofc)                   | Scripts to edit/use |
| ------------------- | ------------------------------------------------------------ | ------------------- |
| i) Tensorflow       | - `*.pb` frozen graph file (protobuf)<br />- `*.pbtxt` a graph configuration file | `tf_*`              |
| ii) Tensorflow Lite | - `*.tflite` lite-converted frozen graph file<br />- `*.pbtxt` labels text file | `tflite_*`          |



The practice established in this repo is that you save these files to the `./tf_files` folder. A prototype whale/dolphin detection model is available [here](https://drive.google.com/open?id=1UQIvXdmQ_rGPZ1nPjrrqs1bi8qNy63sW) so you can get started. Place these files into the `./tf_files`folder and adjust the paths in the scripts if need be. You can also pass these paths via command line, as per the following example `--help` output. All root `./*.py` scripts have this option available.



```shell
$ python tf_cam_detect.py --help
usage: tf_cam_detect.py [-h] [-m GRAPH_PB] [-c GRAPH_CONFIG] [-p MIN_PROB]

optional arguments:
  -h, --help            show this help message and exit
  -m GRAPH_PB, --graph_pb GRAPH_PB
                        path to Tensorflow protobuf object detection model
  -c GRAPH_CONFIG, --graph_config GRAPH_CONFIG
                        path to pbtxt graph config file
  -p MIN_PROB, --min_prob MIN_PROB
                        minimum probability to filter detections
```



## how to train your own model and obtain model files

Besides all you can find on the webs, you can start with this [wiki](https://bitbucket.org/burnpnk/tf-object-detection-viewer/wiki/Home), written with this repo in mind. Also, this project started with this [article from Tony607](https://www.dlology.com/blog/how-to-train-an-object-detection-model-easy-for-free/) (and its associated [repo](https://github.com/Tony607/object_detection_demo)).

