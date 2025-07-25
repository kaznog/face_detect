import face_recognition
import os
import pickle

dataset_path = "faces"

known_encodings = []
known_names = []

# 各人物フォルダを走査
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        print("person_name:"+person_name + " is not dir")
        continue

    print("person_dir:"+person_folder)
    print("person_name:"+person_name)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(image_path)

        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person_name)

# 特徴量を保存
with open("trained_faces.pkl", "wb") as f:
    pickle.dump((known_encodings, known_names), f)

print("顔認証モデルのトレーニングが完了しました。")
