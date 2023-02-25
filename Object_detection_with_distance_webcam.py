import os
import sys
from time import strftime

import numpy as np
import tensorflow as tf
import pytesseract

import cv2
import pyttsx3

if tf.__version__ < '2':
    raise ImportError('Please upgrade your tensorflow installation to v2.2.0 or later!')
if tf.__version__ > '2.10.0':
    raise ImportError('Please downgrade your tensorflow installation to v2.10.0 or earlier!')

sys.path.append("..")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
MIN_SCORE_PRED = 0.60
MIN_SCORE_DIST = 5
MID_X_LOW = 0.3
MID_X_HIGH = 0.7

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Path to model
model_path = 'Object_detection_and_Speech/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model/'
# Load a model from model path
model = tf.saved_model.load(model_path)

# Define prediction function for model using model signature
model_prediction = model.signatures['serving_default']

# Define codec for video output file and frame height, width, fps
fourcc_codec = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = 1280
frame_height = 720
frame_rate = 5

# Initialize webcam feed and set resolution and codec
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
video.set(cv2.CAP_PROP_FOURCC, fourcc_codec)

# Create a video writter to save vide stream
video_stream_save_path = 'Object-detection-and-Speech/video_records'
# Make sure the video save path exists
os.makedirs(video_stream_save_path, exist_ok=True)
# Set video file name according to current time
video_file_name = 'vid_' + strftime("%Y%m%d_%H%M%S") + '.mp4'
video_stream_save_path = os.path.join(video_stream_save_path, video_file_name)
# Initialize video writter with codec and resolution
out = cv2.VideoWriter(video_stream_save_path, fourcc_codec, frame_rate, (frame_height,frame_width))

# initialize pyttsx3
engine =pyttsx3.init()

while(True):
    
    # Acquire frame convert to numpy array
    ret, frame = video.read()
    input_frame = np.asarray(frame)

    # Convert numpy array to tensor
    input_frame = tf.convert_to_tensor(input_frame)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_frame = input_frame[tf.newaxis,...]

    # Perform the actual detection by running the model with the image as input
    model_output = model_prediction(input_frame)

    # extract detection info from model output
    # take only item at 0 index as we sent single image as batch
    num_detections = int(model_output.pop('num_detections'))
    model_output = {key:value[0, :num_detections].numpy() 
                    for key,value in model_output.items()}
    model_output['num_detections'] = num_detections

    # detection_classes should be ints.
    classes = model_output['detection_classes'].astype(np.int64)
    # define detection boxes and scores
    boxes, scores = model_output['detection_boxes'], model_output['detection_scores']

    # Draw the results of the detection (aka 'visulaize the results')
    f,names = vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=MIN_SCORE_PRED)
        
    max_scr_indx = np.argmax(scores)
    text = category_index[classes[max_scr_indx]]['name']
    if text.lower() not in ['person','car','bus','truck']:
        engine.say(text)
        engine.runAndWait()
    # Write frame to video file
    out.write(frame);

    # calculate approximate distance for detected boxes 
    for i,box in enumerate(boxes):
        # skip boxes with score less then threshold
        if scores[i] < MIN_SCORE_PRED:
            continue
        
        # calculate approximate distance
        mid_x = (box[1]+box[3])/2
        mid_y = (box[0]+box[2])/2
        apx_distance = round(((2 - (box[3] - box[1]))**4),1)
        #                 car                    bus                  truck
        if classes[i] == 3 or classes[i] == 6 or classes[i] == 8:

            if apx_distance <= MIN_SCORE_DIST and (mid_x > MID_X_LOW and mid_x < MID_X_HIGH):
                cv2.putText(frame, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                print("Warning -Vehicles Approaching")
                engine.say("Warning -Vehicles Approaching")
                engine.runAndWait()
            else:
                engine.say("Vehicle is at a safer distance")
                engine.runAndWait()
        
        if classes[i] ==44:
            cv2.putText(frame, 'dist', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
            if apx_distance <= MIN_SCORE_DIST and (mid_x > MID_X_LOW and mid_x < MID_X_HIGH):
                cv2.putText(frame, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                print("Warning -Bottle very close to the frame")
                engine.say("Warning -Bottle very close to the frame")
                engine.runAndWait()
            else:
                engine.say("Bottle is at a safer distance")
                engine.runAndWait()

        if classes[i] ==1:
            cv2.putText(frame, 'dist', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
            if apx_distance <= MIN_SCORE_DIST and (mid_x > MID_X_LOW and mid_x < MID_X_HIGH):
                cv2.putText(frame, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                print("Warning -Person very close to the frame")
                engine.say("Warning -Person very close to the frame")
                engine.runAndWait()
            else:
                engine.say("Person is at a safer distance")
                engine.runAndWait()
        
        if cv2.waitKey(25) & 0xFF==ord('q'):
            break
                    
    cv2.imshow('live_detection', frame)
    if cv2.waitKey(25) & 0xFF==ord('q'):
        break
    
#clean everything
cv2.destroyAllWindows()
video.release()
out.release()
exit()