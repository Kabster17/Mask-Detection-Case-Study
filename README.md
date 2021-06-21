# Mask-Detection-Case-Study
Mask Detection using YOLO

## Introduction:
During the coronavirus outbreak, it has become evident that one of most effective preventive measure is to wear a face mask. Given the current situation almost everyone is wearing face masks at all times in public places during the pandemic and in country like Singapore it is now against the law if you were to appear without a mask at public events or places.
This incentivizes us to have the need to explore face mask detection technology to monitor people wearing or not wearing masks in public places. 
Case Study:
With the rise of AI and many machine learning methodologies and approaches, deep learning is gaining popularity in the fields of object detection and classification. Many deep learning frameworks and architectures have been made easily available and adaptable for various use case.
In recent times, many have explored various deep learning approaches to perform Mask Detections and have proven to be successful in their implementations in terms of scalability and accuracy in the detections. One of such state-of-the-art object detection model is known as YOLO. 
In recent times, there are published research in the fields of Mask-Detections using YOLOv3, YOLOv4 and even Faster-RCN. Here are some research paper references implementations using YOLO as the main object detection model to solve the issue of mask detections. 
1.	https://link.springer.com/article/10.1007/s11042-021-10711-8
2.	https://www.mdpi.com/1424-8220/21/9/3263/pdf
3.	https://www.mdpi.com/2079-9292/10/7/837/pdf
The recent for YOLO’s popularity in solving problem statements which is related to object detections and classifications are mainly due to its speed in the detections and its single stage/pipeline to perform the detections. There have been continual work and research being carried on YOLO over the years since its creation since 2016. 
I have decided to train and test a Tiny-YOLOv3 model to check on its detection results and accuracy with my own collected dataset. 


## Methodology:
•	Architecture: 
Tiny-YOLOv3 model (trained using Darknet from AlexeyAB’s Github:  https://github.com/AlexeyAB/darknet
With adjustment made to tweak the hyper parameters catered to a binary classification problem with custom anchors and pre-trained weights.  
Training iterations at 20000.
•	Classes:
2 – Classes (Mask-ON vs Mask-OFF)
•	Data Collection / Annotated:
Self-Collected Data from Google Images: 
100 Images for Mask-ON / 100 Images for Mask-OFF 
Each class images have been augmented at random 4 folds using imgaug library (random rotation/horizontal flip/vertical flip) 
Total Training Images: 1000 (500 for each class)
Images are labelled using labelimg
Mask-ON Labelling
(Labelling is only done on the mask and not bounded to the entire face)
Mask-OFF Labelling
(Labelling is only done on the lower part of frontal part (exposed nose and mouth are the target feature for labelling not the entire face)
•	Testing / Real-Time Detection
Imported the configuration file of tiny-yolov3 and final weights of the model into a python directory 
Created a separate open-cv testing scripts for:
1.	Image Testing – [Mask-Detection-Image.py]
2.	Video/WebCam Testing – [Mask-Detection.py]

## Replicating the Test Scripts
Requirements:
Python and OpenCV 
tiny-yolov3.cfg file – custom configuration file (used for training) 
tiny-yolov3.weights – final trained weights 
obj.names – names file (containing the class names)
Parameters to check and change in the testing scripts:
confThreshold (confidence threshold), nmsThreshold (non-maximum suppression threshold), whT (width and height of the image input)

## Problems – Cause - Solutions

Problem Faced	Cause	Solution
Certain orientations of the face (whether masked or unmasked cannot be detected)	The dataset collected were not as vast or robust to cater to various types of mask orientation.	Collect more data and maybe generate augmentations to boost the data variations further 
Image testing sometimes do not produce all the detections	Due to the different resolutions of the images collected 	Need to standardize the image resolution amongst the training and testing images (without compromising on the quality of the images) 
Need to self-collect images using standard devices. 

The workaround is to set the width and height (whT) to be closer to the test image resolution. 
Detection speed can be improved (more real-time for webcam feed and for the video files if GPU can be utilised)	No GPU available to perform the testing 	Incorporate the GPU usage if possible to check on the detections 
Mask-OFF is essentially a face recognition problem and there might be some bias in the training dataset collected, leading to missed detections.	Humans generally have different facial features and structures which has a lot to do with many factors  (facial hair, accessories, gender, skin tone, age, orientation, expression)	Need to collect data in a more ordered fashion with various segmentations to cover as many variations and differences as possible to allow a more robust training/model. 
Different types of cloth masks (designer/graphics/coloured) type of masks were not being detected well.	Many have started to wear special type of masks and designer masks as a kind of fashion statement. 	Need to collect more data in the varying types of masks (cloth type, designer, multi-coloured, special designed)

Maybe even classify them as a separate mask on category since they are kind of anomaly masks (unique types)



## Future Work
•	Improve on the real-time optimization of the model
•	Gather a vast set of data covering all possible categories of Mask-ON and Mask-OFF features/possibilities
•	Possibility in looking into detecting if a person is wearing the mask correctly or incorrectly as there are instances and cases where people have been caught not wearing the masks properly and exposing their nose/mouth 
Some even use objects and t-shirts as substitute to wearing masks which is not safe/approved and prohibited by law. 

## Directory files:
1.	tiny-yolov3.cfg (config file)
2.	tiny-yolov3.weights (last weights file)
3.	obj.names (class names file)
4.	Mask-Detection.py (script for webcam feed detections)
5.	Mask-Detection-Image.py (script for Image detections)
