import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np
from lxml import etree as ET


plate_prediction = tf.keras.models.load_model('plate_recognition_model.h5')





def plot_bounding_boxes_from_image(images, annotations):
    length = len(images)
    images = images * 255
    annotations = annotations * 255

    for i in range(length):
        image = (images[i]).astype('uint8')
        cv2.rectangle(
            image,
            (int(annotations[i][0]), int(annotations[i][1])),
            (int(annotations[i][2]), int(annotations[i][3])),
            (0, 255, 0),  2 )
       
        cv2.imshow('Image with Bounding Box', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def load_data_from_path(path):
    
    images = []
    for image in path:
        img = cv2.imread(image)
        img = cv2.resize(img, (256, 256))
        images.append(np.array(img))
    return (np.array(images))

def make_prediction_from_image_folder(images_path):
    images = load_data_from_path(images_path)
    images = images/255

    predictions = plate_prediction.predict(images)
    plot_bounding_boxes_from_image(images, predictions)

def make_prediction_from_one_frame(image):
    image = cv2.resize(image, (256, 256))
    image = np.array(image)
    image = image/255
    image = image.reshape(1, 256, 256, 3) 
    prediction = plate_prediction.predict(image)
    
    return prediction

def draw_bounding_box(frame, box):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


def make_prediction_from_video(video_path):

    vid = cv2.VideoCapture(0)

    while(True):
        ret, frame = vid.read()
        frame = cv2.resize(frame, (256, 256))
        if not ret:
            break
        
        
        # Make prediction from the current frame
        prediction = make_prediction_from_one_frame(frame)
        
        # Assuming the prediction is in the format [x1, y1, x2, y2]
        box = prediction[0]  # Get the first prediction
        
        # Rescale the box coordinates back to the original frame size
        height, width, _ = frame.shape
        x1, y1, x2, y2 = box
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        # Draw the bounding box on the frame
        draw_bounding_box(frame, (x1, y1, x2, y2)) 

        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # The 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop, release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

dataset_path = "dataset"
test_images_path = os.path.join(dataset_path, 'test', 'images', '*g')
test_images_path = glob.glob(test_images_path)

make_prediction_from_image_folder(test_images_path)
