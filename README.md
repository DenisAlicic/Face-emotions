# Face-emotions
One solution for Kaggle challenge ["Facial Expression Recognition Challenge"](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
Project for course "Computational Intelligence" - Faculty of Mathematics, University of Belgrade.

### Project idea
Training deep learning model for classification problem of face emotions using pretrained face embeddings from library [face_recogntion](https://pypi.org/project/face_recognition/). 
All details in [paper](https://github.com/DenisAlicic/Face-emotions/blob/master/paper/emotions_recognizer.pdf)

### Using
If you want to try demo, clone repository, install [requirements](https://github.com/DenisAlicic/Face-emotions/blob/master/requirements.txt) and run:
`python predict.py -model PATH_TO_MODEL -save (PATH_TO_FOLDER_FOR_SAVING_PICTURES | 'no') -i ('camera' | PATH_TO_PICTURE)`
