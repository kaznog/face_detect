# 顔認証
import face_recognition
import cv2
import pickle
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import ctypes

# トレーニング済みデータを読み込み
with open("trained_faces.pkl", "rb") as f:
    known_encodings, known_names = pickle.load(f)

# 初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2)

# YOLOv8モデル読み込み（人物検出）
yolo_model = YOLO("yolov8n.pt")  # 軽量版。他に yolov8s.pt, yolov8m.pt, yolov8l.pt あり

# Windows name
window_name = 'YOLOv8 + Face + Pose + Hands'

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv8で人物検出
    results = yolo_model.predict(frame, classes=[0], verbose=False)  # class 0 = person

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for (x1, y1, x2, y2) in boxes:
            # YOLOの検出した人体を線で囲う
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

            roi = frame[y1:y2, x1:x2]
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # 顔認証（ROI限定）
            face_locations = face_recognition.face_locations(rgb_roi)
            face_encodings = face_recognition.face_encodings(rgb_roi, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                name = "Unknown"
                best_match_index = np.argmin(face_distances)
                if face_distances[best_match_index] < 0.6:
                    name = known_names[best_match_index]

                cv2.rectangle(roi, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(roi, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # MediaPipe Pose + Hands （ROI限定）
            roi.flags.writeable = False
            pose_results = pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            hands_results = hands.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            roi.flags.writeable = True

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(roi, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame[y1:y2, x1:x2] = roi  # 処理済みROIを元のframeへ戻す

    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 3)  # -1 = HWND_TOPMOST
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # サイズ変更可能なウィンドウにする
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
pose.close()
hands.close()
