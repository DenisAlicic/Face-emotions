import numpy as np
import tensorflow as tf
from tensorflow import keras
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import argparse
import os

def get_merged(input_folder):

    data = []
    for file in os.listdir(input_folder):
        print(file)
        array = np.load(os.path.join(input_folder,file))
        data.append(array)

    data = np.vstack([data[i] for i in range(0,len(data))])
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
    smote = SMOTE('not majority')
    X_train, Y_train = smote.fit_sample(X_train, Y_train)
    X_train, Y_train = shuffle(X_train, Y_train)
    Y_train = to_categorical(Y_train)

    #Model build
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X_train[0].shape)),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.003)),
        keras.layers.Dense(86, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.002)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(32, activation=tf.nn.tanh),
        keras.layers.Dense(16, activation=tf.nn.relu, bias_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(7, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    # Callback functions
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs_crossentropy_majority_normalized')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.2, patience=10, min_lr=0.001,verbose=1)
    # Run training
    model.fit(X_train, Y_train, epochs=300, callbacks=[tensorboard, reduce_lr], validation_split=0.2)
    # Save model
    if not os.path.exists(args.models_folder):
        os.mkdir(args.models_folder)
    model.save(os.path.join(args.models_folder, "face_emotions_model_crossentropy_majority_normalized.h5"))
if __name__ == "__main__":
    main()
