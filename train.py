import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import preprocess_input
import seaborn as sns

df_train = pd.read_csv('book30-listing-train.csv',encoding = "ISO-8859-1")
df_test = pd.read_csv('book30-listing-test.csv',encoding = "ISO-8859-1")

columns = ['id', 'image', 'link', 'name', 'author', 'class', 'genre']
df_train.columns = columns
df_test.columns = columns
categories = df_train.genre.unique()
classes = df_train['class'].unique()
categories_label = dict()
for i in range(len(categories)):
    categories_label[categories[i]] = classes[i]

df_train.drop_duplicates(subset='name',inplace=True)
df_test.drop_duplicates(subset='name',inplace=True)

image_size = 224
batch_size = 64
epochs = 20

train_datagen = ImageDataGenerator(
                                    rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    validation_split=0.1
                                  )
#usar flow o organizar las categorias en carpetas?
train_generator = train_datagen.flow_from_dataframe(dataframe=df_train, directory='data/train/', class_mode='categorical',
                                                    x_col = 'image', y_col = 'genre',
                                                    batch_size = batch_size, target_size=(image_size,image_size), 
                                                    subset = 'training', shuffle=True, seed=42)
validation_generator = train_datagen.flow_from_dataframe(dataframe=df_train, directory='data/train/', class_mode='categorical',
                                                        x_col = 'image', y_col = 'genre',
                                                        batch_size = batch_size, target_size=(image_size,image_size),
                                                        subset = 'validation', shuffle = True, seed=42)
test_datagen = ImageDataGenerator(rescale=1./255)

xception_model = tf.keras.applications.xception.Xception(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(224, 224, 3),
    pooling='avg',
    classes=30,
    classifier_activation='softmax'
    )
tf.random.set_seed(73)
model = Sequential()
model.add(xception_model)
model.add(Flatten()) 
model.add(Dense(units=2048, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='softmax'))
model.layers[0].trainable=False
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
save_best = ModelCheckpoint(
filepath = 'new_model.hdf5',
verbose=1, save_best_only=True
)
history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // batch_size, validation_data = validation_generator,\
                         validation_steps = validation_generator.samples // batch_size, epochs = epochs, callbacks=[save_best,early_stopping], verbose=2)

