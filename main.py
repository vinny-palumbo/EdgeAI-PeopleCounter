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
import tensorflow as tf
import numpy as np
import time

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from sys import platform

# Get correct params according to the OS
if platform == "darwin":
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')
else:
    CODEC = 0x00000021

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

LABELS_COCO = ["background","person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", 
   "stop sign", "parking meter", "bench", "bird", "cat", "dog", 
   "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
   "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
   "skis", "snowboard", "sports ball", "kite", "baseball bat", 
   "baseball glove", "skateboard", "surfboard", "tennis racket", 
   "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", 
   "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
   "hot dog", "pizza", "donut", "cake", "chair", "couch", 
   "potted plant", "bed", "dining table", "toilet", "tv", "laptop", 
   "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
   "toaster", "sink", "refrigerator", "book", "clock", "vase", 
   "scissors", "teddy bear", "hair drier", "toothbrush"]


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
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-tf", "--use_tensorflow", type=bool, default=False,
                        help="Flag indicating whether to use tensorflow or not. (Using OpenVINO by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def draw_boxes(frame, result, prob_threshold, time_inference):
    '''
    Draw bounding boxes onto the frame.
    '''
    CAP_HEIGHT, CAP_WIDTH, _ = frame.shape
    
    # add inference time text in lower-left corner
    cv2.putText(frame, "Inference time: {:.2f} secs".format(time_inference), (10, CAP_HEIGHT-10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
    
    detections = result[0][0]
    for detection in detections: # Output shape is 1x1x100x7

        conf = detection[2]
        if conf >= prob_threshold:
            # draw detection bbox
            xmin = int(detection[3] * CAP_WIDTH)
            ymin = int(detection[4] * CAP_HEIGHT)
            xmax = int(detection[5] * CAP_WIDTH)
            ymax = int(detection[6] * CAP_HEIGHT)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            
            # draw detection label
            class_idx = int(detection[1])
            if 0 <= class_idx < len(LABELS_COCO):
                class_name = LABELS_COCO[class_idx]
            else:
                class_name = ''
            cv2.putText(frame, "{} {:.2f}".format(class_name, conf), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
            
    return frame


def get_statistics(result, n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, last_detection_timestamp, flag_alert_when_person_leaves, video_timestamp, prob_threshold):
    '''
    Get statistics regarding people on screen, duration they spend on screen, and total people counted
    '''
    TIME_BUFFER = 1.2
    
    # return time on screen when a person has left the scene to update average duration
    time_person_on_screen = None
    
    # get frame's first object detected info
    detections = result[0][0] # Output shape is 1x1x100x7
    first_detection = detections[0]
    first_detection_conf = first_detection[2]
    first_detection_class_idx = int(first_detection[1])
    if 0 <= first_detection_class_idx < len(LABELS_COCO):
        first_detection_class_name = LABELS_COCO[first_detection_class_idx]
    else:
        first_detection_class_name = ''
    
    # update time since last detection
    time_since_last_detection = video_timestamp - last_detection_timestamp    
    
    # if person detected in the frame with confidence
    if first_detection_conf >= prob_threshold and first_detection_class_name == 'person':
    
        # if there hasn't been a detection in more than a second, confirm person entered the scene
        if time_since_last_detection >= TIME_BUFFER:
            print("\nLog: Person entered at {:.2f} seconds.".format(video_timestamp))
            n_in_frame = 1
            n_persons_entered += 1
            timestamp_person_entered = video_timestamp
            # turn on alert to notify when person will leave the scene
            flag_alert_when_person_leaves = True
        
        # return new detection timestamp
        return n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, time_person_on_screen, video_timestamp, flag_alert_when_person_leaves
    
    # if no person detected in the frame
    else:
        
        # if there hasn't been a detection in more than a second, confirm person left the scene about a second ago
        if time_since_last_detection >= TIME_BUFFER and flag_alert_when_person_leaves:
            print("Log: Person left at {:.2f} seconds.".format(video_timestamp-TIME_BUFFER))
            n_in_frame = 0
            n_persons_left += 1
            time_person_on_screen = (video_timestamp-TIME_BUFFER) - timestamp_person_entered
            # mute alert for person leaving until new person detected
            flag_alert_when_person_leaves = False
        
        # return the last detection timestamp until new detection
        return n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, time_person_on_screen, last_detection_timestamp, flag_alert_when_person_leaves
    

def run_tf_inference(image, graph):
    '''
    Run an inference on an image with the tensorflow graph if not using OpenVINO
    '''
    
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['detection_boxes', 'detection_scores','detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)
            
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                 feed_dict={image_tensor: np.expand_dims(image, 0)})
            
            # format output like OpenVINO's IR output (1x1x100x7)
            dummy_first_column = np.zeros((100,1))
            detection_classes_col = output_dict['detection_classes'].transpose((1,0))
            detection_scores_col = output_dict['detection_scores'].transpose((1,0))
            detection_boxes_cols = output_dict['detection_boxes'].transpose((1,2,0))[:,:,-1]
            result = np.hstack((dummy_first_column,
                                detection_classes_col,
                                detection_scores_col,
                                detection_boxes_cols[:,[1,0,3,2]]))
            result = result[np.newaxis, np.newaxis, :, :]
            
    return result
    
    
def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    if args.use_tensorflow:
        
        PATH_TO_FROZEN_GRAPH = args.model + '/frozen_inference_graph.pb'
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
    else: # Use OpenVINO
    
        from inference import Network
        
        # Initialise the class
        infer_network = Network()

        ### TODO: Load the model through `infer_network` ###
        infer_network.load_model(args.model, args.device)
        infer_input_shape = infer_network.get_input_shape()
        infer_input_width, infer_input_height = infer_input_shape[3], infer_input_shape[2]
    
    # Create a flag for single images
    image_flag = False
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        image_flag = True
    
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    CAP_WIDTH = int(cap.get(3))
    CAP_HEIGHT = int(cap.get(4))
    CAP_FPS = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create a video writer for the output video
    if not image_flag:
        out = cv2.VideoWriter('output_video.mp4', CODEC, CAP_FPS, (CAP_WIDTH,CAP_HEIGHT))
    else:
        out = None
    
    ### TODO: Loop until stream is over ###
    frame_counter = 0
    n_in_frame = 0
    n_persons_entered = 0
    n_persons_left = 0
    timestamp_person_entered = 0
    time_people_on_screen = 0
    last_detection_timestamp = 0
    flag_alert_when_person_leaves = False
    while cap.isOpened():
        
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        frame_counter += 1
        video_timestamp = frame_counter / CAP_FPS
        
        if args.use_tensorflow:
            time_start = time.time()
            # Get the results of the tf inference
            result = run_tf_inference(frame, detection_graph)
            time_end = time.time()
        else:
            time_start = time.time()
            ### TODO: Pre-process the image as needed for the Intermediate Representation ###
            p_frame = cv2.resize(frame, (infer_input_width, infer_input_height))
            p_frame = p_frame.transpose((2,0,1))
            p_frame = p_frame.reshape(1, *p_frame.shape)
            
            ### TODO: Start asynchronous inference for specified request ###
            infer_network.exec_net(p_frame)
            
            ### TODO: Wait for the result ###
            if infer_network.wait() == 0:
                ### TODO: Get the results of the inference request ###
                result = infer_network.get_output()
                
            time_end = time.time()
    
        # calculate inference time
        time_inference = time_end - time_start
        
        # add output annotations
        out_frame = draw_boxes(frame, result, args.prob_threshold, time_inference)
        
        ### TODO: Extract any desired stats from the results ###
        n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, time_person_on_screen, last_detection_timestamp, flag_alert_when_person_leaves = get_statistics(result, n_in_frame, n_persons_entered, n_persons_left, timestamp_person_entered, last_detection_timestamp, flag_alert_when_person_leaves, video_timestamp, args.prob_threshold)
        
        ### TODO: Calculate and send relevant information on ###
        ### Topic "person": keys of "count" and "total" ###
        ### Topic "person/duration": key of "duration" ###
        client.publish("person", json.dumps({"count": n_in_frame, "total": n_persons_entered}))
        # if another person left the scene
        if time_person_on_screen:
            # calculate the new average duration on scene
            time_people_on_screen += time_person_on_screen
            time_people_on_screen_avg = time_people_on_screen / n_persons_left
            client.publish("person/duration", json.dumps({"duration": time_people_on_screen_avg}))
            print("Log: Average time people on screen: {:.2f}".format(time_people_on_screen_avg))
        
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(out_frame)
        sys.stdout.flush()
        
        ### TODO: Write out the frame, depending on image or video ###
        if image_flag:
            cv2.imwrite('output_image.jpg', out_frame)
        else:
            out.write(out_frame)
        
        # Break if escape key pressed
        if key_pressed == 27:
            break
    
    # Release the capture and destroy any OpenCV windows
    if not image_flag:
        out.release()
    cap.release()
    cv2.destroyAllWindows()
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
