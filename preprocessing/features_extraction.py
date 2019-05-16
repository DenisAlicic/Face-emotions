import cv2
import os
import face_recognition
import argparse
import numpy as np

def extract_features(input_directory, output_directory, folder, label):
    print(input_directory)
    print(output_directory)
    print(folder)
    print(label)
    input_path = os.path.join(input_directory,folder)
    data = []
    i = 0
    files = os.listdir(input_path)
    for file in files:
        image = face_recognition.load_image_file(os.path.join(input_path, file))
        face_embeddings = face_recognition.face_encodings(image)
        if len(face_embeddings) == 1:
            data.append(face_embeddings[0])
        i += 1
        if i % 1000 == 1:
            print(i,len(files))
    result = np.column_stack([data,[label]*len(data)])
    print(result.shape)
    np.save(os.path.join(output_directory,str(folder)),result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_directory', help="Input folder with data", required=True)
    parser.add_argument('-o', '--output_directory', help="Output folder for features", required=True)
    args = parser.parse_args()
    labels = {
            "Angry" : 0,
            "Disgust" : 1,
            "Fear" : 2,
            "Happy" : 3,
            "Sad" : 4,
            "Surprise" : 5,
            "Neutral" : 6
            }
    folders = os.listdir(args.input_directory)
    if not os.path.exists(args.output_directory):
        os.mkdir(os.path.abspath(args.output_directory))
    for folder in folders:
        extract_features(args.input_directory, args.output_directory, folder, labels[folder])
if __name__ == "__main__":
    main()
