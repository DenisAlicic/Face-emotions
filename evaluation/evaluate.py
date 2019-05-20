import numpy as np
import argparse
import os
import tensorflow.keras as keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random

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
    parser.add_argument('-test',
            '--test_features',
            help="Folder with face embeddings",
            required=True)

    parser.add_argument('-model',
            '--model_path',
            help="Folder with face embeddings",
            required=True)
    args = parser.parse_args()
    # Loading pretrained model
    model = keras.models.load_model(args.model_path)

    X_test, Y_test = get_merged(args.test_features)
    Y_predict = model.predict_classes(X_test)
    print(classification_report(Y_test, Y_predict))
    print(accuracy_score(Y_test, Y_predict))
    print(confusion_matrix(Y_test, Y_predict))


if __name__ == "__main__":
    main()
