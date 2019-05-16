import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import os
import random

def get_merged(input_folder):

    data = []
    for file in os.listdir(input_folder):
        print(file)
        array = np.load(os.path.join(input_folder,file))
        data.append(array)

    data = np.vstack([data[i] for i in range(0,len(data))])
    random.shuffle(data)
    print(data.shape)
    print(np.bincount(np.array(data[:,-1],dtype='int64')))
    return (data[:,0:-1],data[:,-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
            '--input_folder',
            help="Folder with face embeddings",
            required=True)

    parser.add_argument('-models',
            '--models_folder',
            help="Output folder for models",
            required=True)
    args = parser.parse_args()
    X_train, Y_train = get_merged(args.input_folder)


    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train[0].shape)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation=tf.nn.relu,kernel_regularizer=keras.regularizers.l2(0.02), activity_regularizer=keras.regularizers.l2(0.02)),
        keras.layers.Dense(64, activation=tf.nn.relu,kernel_regularizer=keras.regularizers.l2(0.02), activity_regularizer=keras.regularizers.l2(0.02)),
        keras.layers.Dense(32, activation=tf.nn.tanh, bias_regularizer=keras.regularizers.l2(0.02)),
        keras.layers.Dense(14, activation=tf.nn.relu),
        keras.layers.Dense(7, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')
    model.fit(X_train, Y_train,epochs=500,callbacks=[tensorboard],validation_split=0.2)
    if not os.path.exists(args.models_folder):
        os.mkdir(args.models_folder)
    model.save(os.path.join(args.models_folder,"model.h5"))
if __name__ == "__main__":
    main()
