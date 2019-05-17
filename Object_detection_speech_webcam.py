import numpy as np
import os

import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO

from PIL import Image
import cv2
from gtts import gTTS
import pygame

import time
from time import strftime

if tf.__version__ < '1.10.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.10.* or later!')

sys.path.append("..")


from utils import label_map_util

from utils import visualization_utils as vis_util


# What model to use
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


#Load a frozen graph

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
	sess = tf.Session(graph=detection_graph)

#Loading label map

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize webcam feed
video = cv2.VideoCapture(2)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_file = 'vid_' + strftime("%Y%m%d_%H%M%S") + '.avi'
out = cv2.VideoWriter(os.path.join('video_records', video_file),fourcc, 5, (1920,1080))
ret = video.set(3,1920)
ret = video.set(4,1080)

lists = []
lists.clear()
pygame.init()
frame_count=0;

while(True):

	# Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
	# i.e. a single-column array, where each item in the column has the pixel RGB value
	ret, frame = video.read()
	frame_count+=1
	frame_expanded = np.expand_dims(frame, axis=0)

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict={image_tensor: frame_expanded})

	# Draw the results of the detection (aka 'visulaize the results')
	f,names = vis_util.visualize_boxes_and_labels_on_image_array(
		frame,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates=True,
		line_thickness=8,
		min_score_thresh=0.60)

	out.write(frame);
	# All the results have been drawn on the frame, so it's time to display it.
	cv2.imshow('live_detection', f)
	

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
				if not os.path.isfile(os.path.join('sounds', name)):
					tts = gTTS(text=human_string, lang='en')
					tts.save(os.path.join('sounds', name))

				pygame.mixer.music.load(os.path.join('sounds', name))
				pygame.mixer.music.play()
				
		if (frame_count % 30 == 0):
			lists.clear()

	if cv2.waitKey(25) & 0xFF==ord('q'):
		break
            
#clean everything
cv2.destroyAllWindows()
video.release()
sess.close()

