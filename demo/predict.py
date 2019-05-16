import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import face_recognition
import argparse

def main():
    parser = argparse.ArgumentParser(
        prog=__file__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i',
        '--input_path',
        help='Path to the picture',
        required=True
    )
    parser.add_argument(
        '-model',
        '--model_path',
        help='Path to the model',
        required=True
    )
    labels = {
            0 : "Angry",
            1 : "Disgust",
            2 : "Fear",
            3 : "Happy",
            4 : "Sad",
            5 : "Surprise",
            6 : "Neutral"
            }
    args = parser.parse_args()
    img = cv2.imread(args.input_path)
    #frame = img[:, :, ::-1]
    #face_embedding = np.array(face_recognition.face_encodings(frame))
    face_embedding = np.array(face_recognition.face_encodings(img))
    print(face_embedding.shape)
    model = keras.models.load_model(args.model_path)
    predicton = model.predict_classes(face_embedding)
    print(labels[predicton[0]])
    cv2.imshow("Goal",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
