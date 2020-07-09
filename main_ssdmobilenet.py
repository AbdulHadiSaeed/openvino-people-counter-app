"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

import sys
import numpy as np
from random import randint

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60
framewmb = None


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-lt", "--leave_threshold", type=float, default=1,
                        help="Number of seconds threshold that person won't leave the frame in less than them"
                        "(1 sec by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def assess_scene(total_people_count, prev_people_count, prev_enter_duration, current_people_count, duration, args, mqttclient):
    '''
    Assess how many people in current frame and compare with previous frame and publish to MQTT accrodingly
    '''
    enter_duration = prev_enter_duration

    # allow for model inference accuracy issues:
    # in case some people was in previous frame but not detected in current frame by model
    # assuming that people won't instantly leave the frame and need 1 second at least to enter/leave the frame
    if ((duration - prev_enter_duration) < args.leave_threshold):
        if(current_people_count > prev_people_count):
            # return same status to assess next frames
            return total_people_count, prev_people_count, prev_enter_duration
        elif((current_people_count < prev_people_count)): # box blinks
            #return total_people_count, prev_people_count, prev_enter_duration
            mqttclient.publish("person", json.dumps({"count": current_people_count}))
            return total_people_count, current_people_count, duration



    # Check for new people in frame compared to previous frame 
    if(current_people_count > prev_people_count):
        # record the duration they enter the video to calculate their duration in frame later once they leave
        enter_duration = duration

        # send updated people count
        mqttclient.publish("person", json.dumps({"count": current_people_count}))

    # Check for people left the frame compared to previous frame
    elif(current_people_count < prev_people_count):
        # record time they left the frame
        leave_duration = duration
        # calculate duration in frame for left people
        duration_in_frame = leave_duration - prev_enter_duration
        mqttclient.publish("person/duration", json.dumps({"duration": duration_in_frame}))
        # increase total counter for new people only
        total_people_count += 1
        mqttclient.publish("person", json.dumps({"total": total_people_count,"count": current_people_count}))
    
    else:
        # send notificationin case person duration exceeds 10 seconds
        if((duration - prev_enter_duration) == 11 ):
            mqttclient.publish("person/exceeds10sec", json.dumps({"exceeds10sec": "true"}))

    # return current status to assess next frames
    return total_people_count, current_people_count, enter_duration

def draw_boxes(frame, result, args, width, height, infer_time):
    '''
    Draw bounding boxes and inference time onto the frame.
    '''
    current_people_count = 0
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        # box[1] contains class id and 1 means person
        if  conf >= args.prob_threshold and box[1] == 1:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 1)
            cv2.putText(frame,"person", (xmin,ymin+20), 0, 0.6, (255,0,0)) 
            current_people_count += 1

    cv2.putText(frame,"infer time= "+str('{:.1f}'.format(infer_time*1000)) +" ms", (80,40), 0, 0.5, (250,0,0),1)       
    return current_people_count, frame


def check_inputfile(args):
    is_image = False
    # Checks if camera feed and return 0 for opencv to read camera feed
    if args.input == 'CAM':
        return is_image, 0
    # Checks if image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        is_image = True
        return is_image, args.input
    else:#else consider it as video file
         return is_image, args.input


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()

    ### TODO: Handle the input stream ###
    # Get and open video capture
    single_image_mode, inputfile = check_inputfile(args)
    cap = cv2.VideoCapture(inputfile)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    cap.open(inputfile)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # init scene variables
    prev_people_count = 0
    total_people_count = 0
    prev_enter_duration = 0
    frame_count = 0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the image as needed ###
        # Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Start asynchronous inference for specified request ###
        # start time of inference
        start_time = time.time()
        # Perform inference on the frame
        infer_network.exec_net(p_frame)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            # end time of inference
            end_time = time.time()
            frame_count += 1
            ### TODO: Get the results of the inference request ###
            result = (infer_network.get_output())[infer_network.output_blob]
            ### TODO: Extract any desired stats from the results ###
            #inference time
            infer_time = end_time - start_time
            current_people_count, out_frame = draw_boxes(frame, result, args, width, height, infer_time)
                
            #log.info(msg =result.shape)
            #print(result)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            total_people_count, prev_people_count, prev_enter_duration = assess_scene(total_people_count, 
                                            prev_people_count,
                                            prev_enter_duration, 
                                            current_people_count,
                                            frame_count/FPS, # calculate duration for this frame in video
                                            args, client)

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    # Disconnect from MQTT
    client.disconnect()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
