import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import cv2
import os
import glob
from lxml import etree

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D,MaxPooling2D,Dense

import shutil as sh
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from keras.optimizers import Adam
import matplotlib.patches as patches


dataset_path = "dataset"
annotation_path = os.path.join(dataset_path, "annotations")
images_path = os.path.join(dataset_path, "images")

train_images_path = os.path.join(dataset_path, 'train', 'images')
val_images_path = os.path.join(dataset_path, 'val', 'images')
test_images_path = os.path.join(dataset_path, 'test', 'images')

train_annotations_path = os.path.join(dataset_path, 'train', 'annotations')
val_annotations_path = os.path.join(dataset_path, 'val', 'annotations')
test_annotations_path = os.path.join(dataset_path, 'test', 'annotations')

print("Dataset path:", dataset_path)
print("Annotation path:", annotation_path)
print("Images path:", images_path)

# Split images and annotations into train, val, and test sets
def split_images():
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(test_images_path, exist_ok=True)

    os.makedirs(train_annotations_path, exist_ok=True)
    os.makedirs(val_annotations_path, exist_ok=True)
    os.makedirs(test_annotations_path, exist_ok=True)

    images_list = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

    train_files, temp_files = train_test_split(images_list, test_size=0.3, random_state=42)  # 70% training
    val_files, test_files = train_test_split(temp_files, test_size=0.33, random_state=42)  # 20% validation, 10% test

    for file in train_files:
        sh.copy(os.path.join(images_path, file), train_images_path)
        annotation_file = os.path.splitext(file)[0] + '.xml'
        sh.copy(os.path.join(annotation_path, annotation_file), train_annotations_path)

    for file in val_files:
        sh.copy(os.path.join(images_path, file), val_images_path)
        annotation_file = os.path.splitext(file)[0] + '.xml'
        sh.copy(os.path.join(annotation_path, annotation_file), val_annotations_path)

    for file in test_files:
        sh.copy(os.path.join(images_path, file), test_images_path)
        annotation_file = os.path.splitext(file)[0] + '.xml'
        sh.copy(os.path.join(annotation_path, annotation_file), test_annotations_path)

split_images()

# Resize images and annotations
def resize_image_and_annotation(image_path, annotation_path, target_size=(256, 256)):
    # Open the image
    img = Image.open(image_path)
    original_size = img.size  # (width, height)
    img = img.resize(target_size)
    img.save(image_path)

    # Parse the XML file
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Adjust size attributes in the XML
    size = root.find('size')
    size.find('width').text = str(target_size[0])
    size.find('height').text = str(target_size[1])

    # Adjust bounding box coordinates
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')

        xmin.text = str(int(int(xmin.text) * target_size[0] / original_size[0]))
        ymin.text = str(int(int(ymin.text) * target_size[1] / original_size[1]))
        xmax.text = str(int(int(xmax.text) * target_size[0] / original_size[0]))
        ymax.text = str(int(int(ymax.text) * target_size[1] / original_size[1]))

    # Save the updated XML
    tree.write(annotation_path)


# Resize images and annotations for train, val, and test sets
def resize_dataset(images_path, annotations_path, target_size=(256, 256)):
    for img_file in os.listdir(images_path):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
            img_path = os.path.join(images_path, img_file)
            annotation_file = os.path.splitext(img_file)[0] + '.xml'
            annotation_path = os.path.join(annotations_path, annotation_file)
            if os.path.exists(annotation_path):
                resize_image_and_annotation(img_path, annotation_path, target_size)


# Resize all images and annotations
resize_dataset(train_images_path, train_annotations_path, target_size=(256, 256))
resize_dataset(val_images_path, val_annotations_path, target_size=(256, 256))
resize_dataset(test_images_path, test_annotations_path, target_size=(256, 256))


def annotation_to_df(annotation_folder_path):
    images = []
    dictionary = {
        "filename": [], "xmin": [], "ymin": [],
        "xmax": [], "ymax": [], "name": [],
        "width": [], "height": [],
    }

    for annotation in glob.glob(annotation_folder_path + "/*.xml"):
        tree = ET.parse(annotation)
        filename = tree.find('filename').text
        for elem in tree.iter():
            if 'size' in elem.tag:
                for attr in list(elem):
                    if 'width' in attr.tag:
                        width = int(round(float(attr.text)))
                    if 'height' in attr.tag:
                        height = int(round(float(attr.text)))

            if 'object' in elem.tag:
                for attr in list(elem):
                    if 'name' in attr.tag:
                        name = attr.text
                        dictionary['name'] += [name]
                        dictionary['width'] += [width]
                        dictionary['height'] += [height]
                        dictionary['filename'] += [filename]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                xmin = int(round(float(dim.text)))
                                dictionary['xmin'] += [xmin]
                            if 'ymin' in dim.tag:
                                ymin = int(round(float(dim.text)))
                                dictionary['ymin'] += [ymin]
                            if 'xmax' in dim.tag:
                                xmax = int(round(float(dim.text)))
                                dictionary['xmax'] += [xmax]
                            if 'ymax' in dim.tag:
                                ymax = int(round(float(dim.text)))
                                dictionary['ymax'] += [ymax]
    temp_df = pd.DataFrame(dictionary)
    return temp_df


df_training_resized = annotation_to_df(train_annotations_path)
df_val_resized = annotation_to_df(val_annotations_path)
df_test_resized = annotation_to_df(test_annotations_path)




def load_data_from_df(df, images_path):
    images = []
    annotations = []

    for idx, row in df.iterrows():
        img_path = os.path.join(images_path, row['filename'])

        # Open the image and convert to RGB if necessary
        img = cv2.imread(img_path)
        img = np.array(img)
        images.append(img)

        # Normalize bounding box coordinates
        orig_width = row['width']
        orig_height = row['height']

        xmin = row['xmin'] 
        ymin = row['ymin'] 
        xmax = row['xmax'] 
        ymax = row['ymax']


        annotations.append([int(xmax), int(ymax), int(xmin), int(ymin)])

    return np.array(images), np.array(annotations)

training_images, training_annotations = load_data_from_df(df_training_resized, train_images_path)
val_images, val_annotations = load_data_from_df(df_val_resized, val_images_path)
test_images, test_annotations = load_data_from_df(df_test_resized, test_images_path)

training_images = training_images / 255
training_annotations = training_annotations / 255
val_images = val_images / 255
val_annotations = val_annotations / 255
test_images = test_images / 255
test_annotations = test_annotations / 255


def plot_bounding_boxes_from_image(images, annotations):
    length = len(images)
    images = images*255
    annotations = annotations*255
    
    for i in range(length):
        image = cv2.rectangle((images[i]).astype('uint8'),(int(annotations[i][0]),int(annotations[i][1])),(int(annotations[i][2]),int(annotations[i][3])),(0, 255, 0))
        plt.figure()  # create a new figureS
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV uses BGR, matplotlib uses RGB
        plt.show()



model = Sequential()
model.add(Conv2D(64, (3,3), input_shape=(256,256,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.1))

model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.compile(optimizer=Adam(learning_rate=0.0001),  # Adjusted learning rate
              loss='binary_crossentropy',            # Changed loss function
              metrics=['accuracy'])

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)
]

history = model.fit(training_images, training_annotations, validation_data=(val_images, val_annotations), batch_size=16, epochs=30, callbacks=callbacks, verbose=2)

test_loss, test_accuracy = model.evaluate(test_images, test_annotations)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

# Data from training history
history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

epochs = range(1, len(train_loss) + 1)

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

model.save("plate_recognition_model.h5")