import numpy as np
import argparse
import os
from sklearn.utils.multiclass import unique_labels
import tensorflow.keras as keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

def plot_confusion_matrix(y_true, y_pred, classes,
                          title=None,
                          cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Stvarne klase',
           xlabel='Prediktovane klase')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


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
    plot_confusion_matrix(Y_test, Y_predict,classes=['Ljutnja', 'Gadjenje', 'Strah', 'Sreca', 'Tuga', 'Iznenadjenje','Neutralno'],title="Matrica konfuzije")
    plt.show()

if __name__ == "__main__":
    main()
