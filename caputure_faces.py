import cv2
import os
import sys
import glob
import ctypes

# 引数から person_name を取得
if len(sys.argv) != 2:
    print("使い方: python add_faces.py [person_name]")
    sys.exit(1)

person_name = sys.argv[1]
save_dir = f"faces/{person_name}"
os.makedirs(save_dir, exist_ok=True)

# 既存ファイルの数を取得して通番を継続
existing_files = glob.glob(f"{save_dir}/{person_name}_*.jpg")
exist_count = count = len(existing_files)

# 顔検出器の準備
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
window_name = "Add Faces"
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        filename = os.path.join(save_dir, f"{person_name}_{count}.jpg")
        cv2.imwrite(filename, face_img)
        count += 1

        hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
        ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 3)  # -1 = HWND_TOPMOST

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow(window_name, frame)

    # 'q'キーで終了 or 最大枚数に達したら終了
    if cv2.waitKey(1) & 0xFF == ord('q') or (count - exist_count) >= 10:
        break

cap.release()
cv2.destroyAllWindows()