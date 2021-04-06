from reading_datasets import *
import tensorflow as tf
import pandas as pd

target = ds1.pop('Result')
dataset = tf.data.Dataset.from_tensor_slices((ds1.values, target.values))

for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

train_dataset = dataset.shuffle(len(ds1)).batch(1)


def get_compiled_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model


model = get_compiled_model()
model.fit(train_dataset, epochs=15)

