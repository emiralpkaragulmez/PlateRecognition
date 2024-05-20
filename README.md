### License Plate Recognition
# Problem Definiton

Train a model that can predict license plates from images.

# Model Selection

I used cnn model to train my model and my layers are in order:

1. Convolution
2. Max Pooling
3. Dropout
4. Convolution
5. Max Pooling
6. Dropout
7. Flatten
8. Dense
9. Dense

Total Parameters     : 7,893,092
Trainable Parameters : 7,893,092

# Dataset and Preprocces

You can reach data set via link.
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download

Dataset has car images that taken from different angles and .xml files that has anotations of license plates` locations.

1. Spliting data into 3 set.
2. Extracting license plate locations and combining with images
3. Image reshaping.
4. Normalization

# Examples
![image](https://github.com/emiralpkaragulmez/PlateRecognition/assets/79288291/fd7437b3-9001-4476-9bb5-d31e012cecd3)

   
