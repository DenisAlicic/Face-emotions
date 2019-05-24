import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import face_recognition
import argparse
import sys

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
    face_locations = face_recognition.face_locations(img)
    if len(face_locations) == 0:
        print("There is no face in the image")
        sys.exit()
    face_embedding = np.array(face_recognition.face_encodings(img,face_locations))
    model = keras.models.load_model(args.model_path)
    predictons = model.predict_classes(face_embedding)
    img = cv2.resize(img,(240,240))
    text = labels[predictons[0]]
    cv2.imshow(text,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
