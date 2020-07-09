# Project Write-Up

## Comparing Model Performance
### IR Model [Yolov3 tiny FP16]
 In this github repo under `model/frozen_darknet_yolov3_model_tiny_fp16.xml`
### Model Origin
I used and converted tensorflow yolov3 model mentioned in [this guide](https://docs.openvinotoolkit.org/2020.2/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html) which is in this [github repo](https://github.com/mystic123/tensorflow-yolo-v3).

I've used Yolov3 sample from openvino toolkit to parse the model output  and I have made some modifications to it to make suitable for app. It's `yolov3_helper.py` in project root.

I've used `demo.py` from same Yolov3 [github repo](https://github.com/mystic123/tensorflow-yolo-v3) to run inference using the original frozen model and I have made some modifications to it to get video output as well as print inference time. It's `tf_yolov3_demo.py` in project root.

### Method
My method(s) to compare models before and after conversion to Intermediate Representations were to to create output video of sample video with bounding boxes and inference time printed for pre-conversted model and then watch the video and discover the false positive/false negatives/inference time based on it.  


| Factor/Model       | YOLOV3 Tiny   | YOLOV3        | SSD MobileNetV2 | 
|--------------------|---------------|---------------|---------------|
|Pre-Size            | 34MB          | 242MB         |  68MB         |
|Post-Size FP32      | 34MB          | 242MB         |  65MB         |   
|Post-Size FP16      | 17MB          | 120MB         |  32MB   |   
|||||   
|Pre-Inference Time  | 55ms      | 440ms         |   not tested |
|Post-Inference Time FP32| 15ms      | 170ms     |  12ms |
|Post-Inference Time FP16| 15ms      | 170ms     | 12ms|
|||||
|Pre-Accuracy  with **0.5** threshold | True Negative = low | 100% Accurate  | not tested |
|Post-Accuracy FP32  with **0.2** threshold| False Positive = low <br /> False Negative = low<br /> | False Positive =0  <br /> False Negative = low  |  False Positive =low  <br /> False Negative = **high** |
|Post-Accuracy FP16 with **0.2** threshold| False Positive = low <br /> False Negative = low<br /> | False Positive =0  <br /> False Negative = low  |  False Positive =low  <br /> False Negative = **high** | 
|Does it fullfil requirements?| YES and fast| YES but slow| NO |
|Selected Model| **Yolov3 tiny FP16** | | |

## Assess Model Use Cases

Some of the potential use cases of the people counter app are
* Monitor people entering/leaving/duration at specifc location (office/shop/goverment entity) to check peak hours and may give some alerts once specifc count/duration exceeded
* Monitor clients durations at customer service kiosks and enhance the customer service process (SLA/kiosks location etc..)
* Monitor customer on shop and how much time the spent in front each location in the shop and try to enhance the locations where customer only spent few seconds (not attactive goods/design) 

Each of these use cases would be useful because they will automatically visually monitor required spaces all around the clock with ability to give alerts on the spot to take actions if needed.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows
* Lighting: I have made the sample video light as dark and run the inference on it (added to this repo under `resources/Pedestrian_Detect_2_1_1_n.mp4`), the true negatives are increased and the bounding boxes where smaller and doesn't fit the person. That means it will impact accurcy of the model.
* Accuracy: model accuracy should be enough to fullfil requirements for example in this app +/-1 person count is allowed and that impact selecting model with fit accuracy. While in a security app, accurcy is much more important so true negative are not accepted.
* focal length/image size: this sould be preconfigured and agreed with customer before selecting the best model and considering the network requirements for image size. 

## MVP Features
I have implemented those features from MVP mentioned features
* Notification in case current person duration exceeds 10 sec.
  * added toast react library to package.json
  * publish new topic to mqtt once duration of current person exceeds 10 seconds
* Add button to stop/start ffmpeg feed
  * added video icon at navigation to show/hide ffmpeg feed
  * set img src to empty in case user select to hide the feed
  * I have monitored the sample video ffmpeg feed sent to UI size and it was about **50MB** eventhough the original video is **3MB**

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Yolov3_Tiny]
  - https://github.com/mystic123/tensorflow-yolo-v3
  - I converted the model to an Intermediate Representation with the following arguments
    ```
    python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny

    export MO_PATH=/opt/intel/openvino/deployment_tools/model_optimizer

    python3 $MO_PATH/mo_tf.py --reverse_input_channels --transformations_config=$MO_PATH/extensions/front/tf/yolo_v3_tiny.json --input_model=frozen_darknet_yolov3_model_tiny.pb --data_type=FP16 --batch 1
    ```
  - The model was **sufficient** 
  - I tried to improve the model for the app by decrease
    - Tune confidence threshold (0.1 and 0.2 works good with yolov3 tiny)
    - Add person enter/leave duration threshold (default 1 second) for considering that person won't enter and leave under that threshold. This will remove the impact of few ture negative and false positives
    - Make precision as 16FP to reduce size of the model
  
- Model 1: [Yolov3]
  - https://github.com/mystic123/tensorflow-yolo-v3
  - I converted the model to an Intermediate Representation with the following arguments
    ```
    python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights 

    export MO_PATH=/opt/intel/openvino/deployment_tools/model_optimizer

    python3 $MO_PATH/mo_tf.py --reverse_input_channels --transformations_config=$MO_PATH/extensions/front/tf/yolo_v3.json --input_model=frozen_darknet_yolov3_model.pb --data_type=FP16 --batch 1
    ```
  - The model was **sufficient** but slow
  - I tried to improve the model for the app by 
    - Tune confidence threshold (0.1 and 0.2 works good with yolov3)
    - Add person enter/leave duration threshold (default 1 second) for considering that person won't enter and leave under that threshold. This will remove the impact of few ture negative and false positives
    - Make precision as 16FP to reduce size of the model

- Model 1: [SSDMobileNetV2]
  - Source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
  - Download and extract from: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
  - I converted the model to an Intermediate Representation with the following arguments
    ```
    export MO_PATH=/opt/intel/openvino/deployment_tools/model_optimizer

    python3 $MO_PATH/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --transformations_config=$MO_PATH/front/tf/ssd_v2_support.json
    ```
  - The model was **insufficient** 
  - I tried to improve the model for the app by 
    - Tune confidence threshold (even with low value 0.1 and 0.2 it doesn't work)
    - Add person enter/leave duration threshold (default 1 second) for considering that person won't enter and leave under that threshold. This will remove the impact of few ture negative and false positives
    - Make precision as 16FP to reduce size of the model


## Explaining Custom Layers

Model Optimizer - while converting model to IR - will classify any model layer not in supported layers list as a custom layer.

The process behind converting custom layers involves:
* Using `extension generator` to generate extensions for both model optimizer as well as inference engine for different targeted hardware and regiter those extensions accordingly.
* Some other options available depending on the original model framework:
  * For Caffe, 2nd option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer.
  * For TensorFlow, 
    * 2nd option is to actually replace the unsupported subgraph with a different subgraph.
    * 3rd option is to actually offload the computation of the subgraph back to TensorFlow during inference. 
