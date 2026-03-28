import tensorflow as tf
import numpy as np

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

def load_direct():
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    return model

print("Testing direct load:")
try:
    m1 = load_direct()
    val1 = m1.predict(np.zeros((1, 128, 128, 3)))
    print("M1 pred shape:", val1.shape)
    print("M1 pred sum:", np.sum(val1))
    print("M1 pred max:", np.max(val1))
except Exception as e:
    print("Direct load error:", e)

print("Testing manual load:")
try:
    m2 = manual_model()
    val2 = m2.predict(np.zeros((1, 128, 128, 3)))
    print("M2 pred shape:", val2.shape)
    print("M2 pred sum:", np.sum(val2))
    print("M2 pred max:", np.max(val2))
except Exception as e:
    print("Manual load error:", e)
