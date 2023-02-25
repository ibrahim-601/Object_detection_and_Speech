# Object-detection-and-Speech
Live object detection using tensorflow object detection api and speech output using gtss and pygame.
## Summary
This repository is a application of tensorflow object detection api to detect objects in webcam feed and 
it gives audible output for the detected object's class name. For audio output is uses google text to speech 
to get audio files for class names and pygame to play the audio.

## Introduction
The repository is tested with Windows 10, and it will also work for Windows 7 and 8. The general procedure can also 
be used for Linux operating systems, but some minor changes might be required.

## Steps to run
The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several additional Python packages, and a few extra setup commands to get everything set up to run or train an object detection model. 

### 1. Environment
Create a virtual environment using anaconda and in the environment install below packages. Or you can use pip.

```
pillow
lxml
Cython
jupyter
matplotlib
pandas
gtts
pygame
pyttsx3
tensorflow >= 2.2
opencv-python >= 4.0
protobuf >= 3.1
```


### 2. Download TensorFlow Object Detection API repository from GitHub
Clone TensorFlow object detection repository located at https://github.com/tensorflow/models or download as zip and extract. Go to the directory where the repo is clone or extracted and navigate to `research/`. Open a terminal in the `research` directory and activate the environment created in [step 1](#1-environment). Run below command to generate python scripts from protocol buffer object present in the `object_detection/protos`
```bash
protoc object_detection/protos/*.proto --python_out=.
```

### 3. Download this repository from GitHub
Clone this repository or download and extract all the contents directly into the `research/object_detection` directory. Replace `object_detection/utils/visualization_utils.py` file with the one found in `Object_detection_and_Speech/utils`. The `Object_detection_and_Speech/utils/visualization_utils.py` file contains some modification required for this project.

### 4. Run the project
From the `object_detection` directory open a terminal and activate environment created in [step 1](#1-environment). Then run the below command:
```bash
python Object_detection_and_Speech/Object_detection_speech_webcam.py
```
If you want to run object detection with distance warning then:
```bash
python Object_detection_and_Speech/Object_detection_with_distance_webcam.py
```