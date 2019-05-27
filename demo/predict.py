import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import face_recognition
import argparse
import sys
import os

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
    parser.add_argument(
        '-save',
        '--save_path',
        help='Path to the save dir,or no',
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
    # Parameters for font 
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    fontColor = (255,0,0)
    lineType = 2

    if args.input_path == "camera":
        camera = cv2.VideoCapture(0)
        while True:
            ret_val, img = camera.read()
            cv2.imshow("Slikaj se!",img)
            if not ret_val:
                sys.exit()
            key = cv2.waitKey(16)
            if key & 0xFF == ord(' '):
                camera.release()
                cv2.destroyAllWindows()
                break
            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                camera.release()
                break
    else:
        img = cv2.imread(args.input_path)
    face_locations = face_recognition.face_locations(img)
    if len(face_locations) == 0:
        print("There is no face in the image")
        sys.exit()
    face_embedding = np.array(face_recognition.face_encodings(img,face_locations))
    model = keras.models.load_model(args.model_path)
    predictons = model.predict_classes(face_embedding)

    text = labels[predictons[0]]

    if args.input_path != "camera":
        img = cv2.resize(img,(240,240))
    else:
        text_width, text_height = cv2.getTextSize(text,font,fontScale,lineType)[0]
        topRightCornerOfText = (img.shape[0]+120,text_height+10)
        cv2.putText(img,text,
                    topRightCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        face_location = face_locations[0]
        cv2.rectangle(img,(face_location[3],face_location[0]),(face_location[1],face_location[2]),fontColor,lineType)
    cv2.imshow(text,img)
    # Saving photo only if camera is turned on
    if args.save_path != "no" and args.input_path == "camera":
        if os.path.exists(os.path.join(args.save_path,os.getlogin() +"_is_"+ text + ".jpg")):
            counter = 0
            while os.path.exists(os.path.join(args.save_path,os.getlogin() +"_is_"+ text + str(counter)+".jpg")):
                    counter += 1
            cv2.imwrite(os.path.join(args.save_path,os.getlogin() +"_is_"+ text + str(counter) +".jpg"), img)
        else:
            cv2.imwrite(os.path.join(args.save_path,os.getlogin() +"_is_"+ text + ".jpg"), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
