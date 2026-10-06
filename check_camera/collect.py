import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

OUT_DIR = "my_dataset"
PARTICIPANT_ID = 1     # đổi theo từng người quay

CHORDS = [
    "A", "Am", "C", "Cm", "D", "Dm", "E", "Em", "F", "G", "none"
]
       
chord = input("Nhập hợp âm (A, Am, C, Cm,...):").strip()
if chord not in CHORDS:
    raise ValueError(f"Hợp âm '{chord}' không hợp lệ!")

SAVE_DIR = os.path.join(OUT_DIR, chord)
os.makedirs(SAVE_DIR, exist_ok = True)
CSV_PATH = os.path.join(OUT_DIR, "train.csv")

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def to_arr(lms, n): #chuyển landmark của MediaPipe thành mảng numpy
    if lms is None:
        return np.full((n, 3), np.nan, dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)

def extract_frame(res):
    return np.concatenate([
        to_arr(res.left_hand_landmarks, 21),
        to_arr(res.right_hand_landmarks, 21),
    ])  

def build_columns():
    columns = []

    for hand in ['left_hand', 'right_hand']:
        for k in range(21):
            for a in 'xyz':
                columns.append(f"{a}_{hand}_{k}")

    return columns

COLUMNS = build_columns()

def save_sequence(frames, chord, file_id, sequence_id):
    data = np.stack(frames)                  
    T = data.shape[0]
    flat = np.concatenate([data[:, :, 0], data[:, :, 1], data[:, :, 2]], axis=1)
    df = pd.DataFrame(flat, columns=COLUMNS)
    df.insert(0, "frame", range(T))
    df.index = [sequence_id] * T
    df.index.name = "sequence_id"
    rel_path = f"{chord}/{file_id}.parquet"
    save_path = os.path.join(OUT_DIR, rel_path)
    df.to_parquet(save_path)

    row = pd.DataFrame([{  #lưu thông tin vào csv
        "path": rel_path, "file_id": file_id, "sequence_id": sequence_id,
        "participant_id": PARTICIPANT_ID, "chord": chord,
    }])
    row.to_csv(CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)
    print(f"Đã lưu {T} frame | '{chord}' | file_id={file_id}")

def next_id():
    if os.path.exists(CSV_PATH):
        return int(pd.read_csv(CSV_PATH)["file_id"].max()) + 1
    return 1000001

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Camera mở được:", cap.isOpened())
for _ in range(30):      # bỏ vài frame đen đầu tiên
    cap.read()
recording, frames = False, []

LEFT_SPEC  = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3)   # xanh dương (BGR)
LEFT_LINE  = mp_draw.DrawingSpec(color=(255, 150, 0), thickness=2)
RIGHT_SPEC = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)   # đỏ
RIGHT_LINE = mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # đọc frame từ camera
        ok, img = cap.read()
        if not ok:
            break
        # lât ảnh ngang
        img = cv2.flip(img, 1)
        # BGR -> RGB để đưa vào MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # MediaPipe nhận diện landmark
        res = holistic.process(rgb)
        # Vẽ landmark tay trái và tay phải
        mp_draw.draw_landmarks(img, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, LEFT_SPEC, LEFT_LINE)
        mp_draw.draw_landmarks(img, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, RIGHT_SPEC, RIGHT_LINE)

        if recording:
            # lấy landmark của 2 tay ở frame hiện tại 
            frames.append(extract_frame(res))
            # hiển thị số frame đã ghi
            cv2.putText(img, f"REC {len(frames)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # hiển thị chord đang thu
        cv2.putText(img, f"Chord: {chord}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # hiển thị camera
        cv2.imshow("Thu du lieu", img)
        # đọc phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if recording and frames:
                fid = next_id()
                save_sequence(frames, chord, fid, fid * 1000 + 1)
                frames = []
                print(f"đã lưu, ấn space để quay tiếp.")
            recording = not recording
        elif key == ord("q"):
            break
# đóng camera
cap.release()
cv2.destroyAllWindows()