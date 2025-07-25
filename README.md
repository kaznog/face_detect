# face_detect

face recognition YOLO detect MediaPipe hands and pose recognition

It must be referenced either by CMake or through a path.

 📸 1. Capture the face of the person to be trained
 ```
    python .\capture_faces.py [person's name]  
 ```
    - If you want to train multiple people, run the command separately for each individual, specifying their name.
    - Photos will be saved under the faces directory, organized into subfolders named after each person.
    - Each capture session saves 10 images.
    If you need higher accuracy, repeat the capture process to double the training data to 20 images.

🧠 2. Create the training dictionary using the captured photos
```
python .\make_pkl.py  
```
✅ 3. Verify recognition using the trained model
```
python .\frecognizer.py  
```
