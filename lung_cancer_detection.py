import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn import metrics

from zipfile import ZipFile
import cv2
import gc
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


import warnings
warnings.filterwarnings('ignore')



#Visualizing the data
path = './lung_colon_image_set/lung_image_sets'

classes = ["lung_n","lung_aca","lung_scc"]#Normal and 2 types of lung cancers

'''
It selects a random sample of three images from each category and visualizes them using Matplotlib.
PIL.Image.open(): open the images and convert them in a format that can be displayed.
'''
for cat in classes:
    image_dir = f'{path}/{cat}'
    images = os.listdir(image_dir)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category . . . .', fontsize=20)

    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(f'{path}/{cat}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')
    #plt.show()

'''
1. Resizing the images and converting them into NumPy arrays for efficient computation.
Since large images are computationally expensive to process we resize them to a standard size (256x256) using 
numpy array. We used 10 epochs with batch size of 64.

2. One hot encoding: Labels (Y) are converted to one-hot encoded vectors using pd.get_dummies(). 
This allows the model to output soft probabilities for each class.

3. Train-Test Split: We split the dataset into training and validation sets i.e., 80% for training and 20% for 
validation. This allows us to evaluate the model's performance on unseen data.
'''

IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 64

X = []
Y = []

for i, cat in enumerate(classes):
  images = glob(f'{path}/{cat}/*.jpeg')

  for image in images:
    img = cv2.imread(image)

    X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    Y.append(i)

X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values
#X_train, X_val, Y_train, Y_val = train_test_split(X, one_hot_encoded_Y, test_size=SPLIT, random_state=2022)
#Splitting into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, one_hot_encoded_Y, test_size=SPLIT, random_state=2022)
#Splitting X_train and Y_train again to keep 10% of data for testing.
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=2022)


#Building the CNN
model = keras.models.Sequential([
    layers.Conv2D(filters=32,
                  kernel_size=(5, 5),
                  activation='relu',
                  input_shape=(IMG_SIZE,
                               IMG_SIZE,
                               3),
                  padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(filters=64,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(filters=128,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')
])
model.summary()


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if logs.get('val_accuracy', 0) > 0.90:
            print('\n Validation accuracy has reached upto \
                      90% so, stopping further training.')
            self.model.stop_training = True


es = EarlyStopping(patience=3,
                   monitor='val_accuracy',
                   restore_best_weights=True)

lr = ReduceLROnPlateau(monitor='val_loss',
                       patience=2,
                       factor=0.5,
                       verbose=1)

#Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',   # or 'sparse_categorical_crossentropy'
    metrics=['accuracy']
)

#Training
history = model.fit(X_train, Y_train,
                    validation_data = (X_val, Y_val),
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS,
                    verbose = 1,
                    callbacks = [es, lr, myCallback()])

#Visualise training and validation accuracy
history_df = pd.DataFrame(history.history)
history_df.loc[:,['accuracy','val_accuracy']].plot()
plt.show()

#Model evaluation using validation set
Y_pred = model.predict(X_val)
Y_val = np.argmax(Y_val, axis=1) #The model outputs probabilities for each class per input. Need to find the 
                                    #class with the highest probability
Y_pred = np.argmax(Y_pred, axis=1)
print(metrics.classification_report(Y_val, Y_pred,
                                    target_names=classes))

#Model evaluation using test set
Y_pred = model.predict(X_test)
Y_test = np.argmax(Y_test, axis=1) #The model outputs probabilities for each class per input. Need to find the 
                                    #class with the highest probability
Y_pred = np.argmax(Y_pred, axis=1)
print(metrics.classification_report(Y_test, Y_pred,
                                    target_names=classes))