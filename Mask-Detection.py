import cv2
import numpy as np

#Parameters to change for detection variable
whT = 608 # width and height of input image
confThreshold = 0.8 # confidence threshold for detections
nmsThreshold = 0.4 # nms threshold
modelConfiguration = 'tiny-yolov3.cfg'  #path to the model config file
modelWeights = 'tiny-yolov3.weights' #path to the model weights


#Reading the YOLO network/architecture from the custom config and weights file
#assigning it to the network
net = cv2.dnn.readNet(modelWeights, modelConfiguration)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# Set preference target for computation (CPU or GPU(CUDA))
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

classes = [] #list to append the classes names (ie. Mask-0N / Mask-OFF)
with open('obj.names', 'r') as f:
    classes = f.read().splitlines()

# Intialising the WebCam video capture
cap = cv2.VideoCapture(0)  # replace (0) with the video file as ('test.mp4') to execute video file detections


while True:
    _, img = cap.read()
    height, width, _ = img.shape

    #Normalising the image and swapping the BGR image to RGB order and assigning
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), (0,0,0), swapRB=True, crop=False)

    net.setInput(blob) #setting the input from blob into the network
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names) #extracting the outputs from the output layers of the network

    boxes = [] #list to store the bounding boxes
    confidences = [] #list to store the confidence score - predicted
    class_ids = [] # list to store the predicted classes in the target area

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)  #extracting the output with the highest confidence score
            confidence = scores[class_id]
            if confidence > confThreshold:            # setting the detection thresholds levels
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    # Non-mAX suppression to limit the number of bounding boxes created for single detection
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

    # setting the label fonts and bounding boxes colors
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))


    if len(indexes) > 0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors [i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)  # 'Esc key to exit the webcam feed buffer'
    if key==27:
        break

cap.release()
cv2.destroyALLWindows()