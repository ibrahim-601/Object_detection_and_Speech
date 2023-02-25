import os
import sys
from time import strftime

import numpy as np
import tensorflow as tf

import cv2
from gtts import gTTS
import pygame

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

# Initialize variables required for frame processing
lists = []
lists.clear()
frame_count=0;

# Initialize pygame
pygame.init()

# loop for detecting object on each frame from webcam
while(True):

	# Acquire frame convert to numpy array
	ret, frame = video.read()
	frame_count+=1
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
	model_output['detection_classes'] = model_output['detection_classes'].astype(np.int64)

	# Draw the results of the detection (aka 'visulaize the results')
	f,names = vis_util.visualize_boxes_and_labels_on_image_array(
		frame,
		model_output['detection_boxes'],
		model_output['detection_classes'],
		model_output['detection_scores'],
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=0.60)

	# Write frame to video file
	out.write(frame);

	# All the results have been drawn on the framerame, so it's time to display it.
	cv2.imshow('live_detection', frame)
	
	# Provide sound output after 5 frames
	if (frame_count % 5 == 0):
		for i in names:
			
			if i not in lists:
				lists.append(i)

				human_string = i

				lst = human_string.split()
				human_string = " ".join(lst[0:2])
				human_string_filename = str(lst[0])

				# Speech module
				if pygame.mixer.music.get_busy() == False and human_string == i:
					name = human_string_filename + ".mp3"
					
				# Only get from google if we dont have it
				if not os.path.isfile(os.path.join('Object-detection-and-Speech/sounds', name)):
					tts = gTTS(text=human_string, lang='en')
					tts.save(os.path.join('Object-detection-and-Speech/sounds', name))

				# Use pygame mixer to play the sound file
				pygame.mixer.music.load(os.path.join('Object-detection-and-Speech/sounds', name))
				pygame.mixer.music.play()
		
		# Clear detected object list after 30 frames
		if (frame_count % 30 == 0):
			lists.clear()

	# Add functionalities to quit the program by pressing 'q' on keyboard
	if cv2.waitKey(25) & 0xFF==ord('q'):
		break
            
#clean everything
cv2.destroyAllWindows()
video.release()
out.release()
exit()
