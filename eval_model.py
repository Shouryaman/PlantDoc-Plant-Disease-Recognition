import tensorflow as tf
import numpy as np
import os

def manual_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[128,128,3]))
    model.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,padding='same',activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1500,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(units=38,activation='softmax'))
    model.load_weights("trained_plant_disease_model.keras")
    return model

model = manual_model()

validation_set = tf.keras.utils.image_dataset_from_directory(
    'valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
val_loss, val_acc = model.evaluate(validation_set)
print(f"Validation accuracy: {val_acc}")
