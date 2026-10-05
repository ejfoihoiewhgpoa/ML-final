import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

OUT_DIR = "my_dataset"
PARTICIPANT_ID = 1            # đổi theo từng người quay
os.makedirs(f"{OUT_DIR}/train_landmarks", exist_ok=True)
CSV_PATH = f"{OUT_DIR}/train.csv"

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def to_arr(lms, n):
    if lms is None:
        return np.full((n, 3), np.nan, dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)

def extract_frame(res):
    return np.concatenate([
        to_arr(res.face_landmarks, 468),
        to_arr(res.left_hand_landmarks, 21),
        to_arr(res.pose_landmarks, 33),
        to_arr(res.right_hand_landmarks, 21),
    ])  # (543, 3)

def build_columns():
    types = ['face']*468 + ['left_hand']*21 + ['pose']*33 + ['right_hand']*21
    idx = list(range(468)) + list(range(21)) + list(range(33)) + list(range(21))
    return [f"{a}_{t}_{k}" for a in "xyz" for t, k in zip(types, idx)]

COLUMNS = build_columns()

def save_sequence(frames, phrase, file_id, sequence_id):
    data = np.stack(frames)                       # (T, 543, 3)
    T = data.shape[0]
    flat = np.concatenate([data[:, :, 0], data[:, :, 1], data[:, :, 2]], axis=1)
    df = pd.DataFrame(flat, columns=COLUMNS)
    df.insert(0, "frame", range(T))
    df.index = [sequence_id] * T
    df.index.name = "sequence_id"
    rel_path = f"train_landmarks/{file_id}.parquet"
    df.to_parquet(f"{OUT_DIR}/{rel_path}")

    row = pd.DataFrame([{
        "path": rel_path, "file_id": file_id, "sequence_id": sequence_id,
        "participant_id": PARTICIPANT_ID, "phrase": phrase,
    }])
    row.to_csv(CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)
    print(f"Đã lưu {T} frame | '{phrase}' | file_id={file_id}")

def next_id():
    if os.path.exists(CSV_PATH):
        return int(pd.read_csv(CSV_PATH)["file_id"].max()) + 1
    return 1000001

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Camera mở được:", cap.isOpened())
for _ in range(30):      # bỏ vài frame đen đầu tiên
    cap.read()
recording, frames = False, []
phrase = input("Nhập câu cần ký hiệu: ").strip()

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.flip(img, 1)
        res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        LEFT_SPEC  = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3)   # xanh dương (BGR)
        LEFT_LINE  = mp_draw.DrawingSpec(color=(255, 150, 0), thickness=2)
        RIGHT_SPEC = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)   # đỏ
        RIGHT_LINE = mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)
        mp_draw.draw_landmarks(img, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                       LEFT_SPEC, LEFT_LINE)
        mp_draw.draw_landmarks(img, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                       RIGHT_SPEC, RIGHT_LINE)
        if recording:
            frames.append(extract_frame(res))
            cv2.putText(img, f"REC {len(frames)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, phrase, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Thu du lieu", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if recording and frames:
                fid = next_id()
                save_sequence(frames, phrase, fid, fid * 1000 + 1)
                frames = []
                phrase = input("Câu tiếp theo (Enter để giữ nguyên): ").strip() or phrase
            recording = not recording
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()