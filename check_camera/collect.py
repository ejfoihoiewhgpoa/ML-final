import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

OUT_DIR = "my_dataset/train_landmarks"
PARTICIPANT_ID = 1     # đổi theo từng người quay

ROOTS = [
    "A", "Am", "C", "Cm", "D", "Dm", "E", "Em", "F", "G", "none"
]

QUALITIES = ["R", "1", "M", "D", "m1"
]     
root = input(f"Nhập hợp âm tay trái: ").strip()
quality = input(f"Nhập loại tay phải: ").strip()
if root not in ROOTS:
    raise ValueError(
        f"Hợp âm '{root}' không hợp lệ!\n"
        f"họn một trong: {ROOTS}"
    )
if quality not in QUALITIES:
    raise ValueError(
        f"Loại '{quality}' không hợp lệ!\n"
        f"họn một trong: {QUALITIES}"
    )

label = f"{root}_{quality}"

#tạo thư mục lưu dữ liệu
SAVE_DIR = os.path.join(OUT_DIR, label)
os.makedirs(SAVE_DIR, exist_ok = True)
CSV_PATH = os.path.join(OUT_DIR, "train.csv")

print(f"\nĐang thu: {label}")
print(f"Tay trái : {root}")
print(f"Tay phải : {quality}")
print(f"Lưu vào  : {SAVE_DIR}\n")

# mediapipe
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

def to_arr(lms, n): #chuyển landmark của MediaPipe thành mảng numpy
    if lms is None:
        return np.full((n, 3), np.nan, dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32)
# lấy landmark của 1 frame
def extract_frame(res):
    return np.concatenate([
        to_arr(res.left_hand_landmarks, 21),
        to_arr(res.right_hand_landmarks, 21),
    ])  
#tạo tên cột
def build_columns():
    columns = []

    for hand in ['left_hand', 'right_hand']:
        for k in range(21):
            for a in 'xyz':
                columns.append(f"{a}_{hand}_{k}")

    return columns

COLUMNS = build_columns()
# lưu 1 sequence
def save_sequence(frames, root, quality, label, file_id, sequence_id):
    data = np.stack(frames)                  
    T = data.shape[0]
    flat = data.reshape(T, -1)
    df = pd.DataFrame(flat, columns=COLUMNS)
    df.insert(0, "frame", range(T))
    df.index = [sequence_id] * T
    df.index.name = "sequence_id"
    rel_path = f"{label}/{file_id}.parquet"
    save_path = os.path.join(OUT_DIR, rel_path)
    df.to_parquet(save_path)

    row = pd.DataFrame([{  #lưu thông tin vào csv
        "path": rel_path, "file_id": file_id, "sequence_id": sequence_id,
        "participant_id": PARTICIPANT_ID, "root": root, "quality": quality, "label": label
    }])
    row.to_csv(CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)
    print(f"Đã lưu {T} frame | '{label}' | file_id={file_id}")

def next_id():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if len(df) > 0:
            return int(pd.read_csv(CSV_PATH)["file_id"].max()) + 1
    return 1000001
# mở camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Camera mở được:", cap.isOpened())
for _ in range(30):      # bỏ vài frame đen đầu tiên
    cap.read()
recording, frames = False, []

# màu vẽ landmark
LEFT_SPEC  = mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3)   # xanh dương (BGR)
LEFT_LINE  = mp_draw.DrawingSpec(color=(255, 150, 0), thickness=2)
RIGHT_SPEC = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)   # đỏ
RIGHT_LINE = mp_draw.DrawingSpec(color=(0, 200, 0), thickness=2)

# chạy mediapipe
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
        # hiển thị nhãn
        cv2.putText(img, f"Chord: {root}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Quality: {quality}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Label: {label}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # hiển thị camera
        cv2.imshow("Thu du lieu", img)
        # đọc phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            if recording and frames:
                fid = next_id()
                sequence_id = fid * 1000 + 1
                save_sequence(frames, root, quality, label, fid, sequence_id)
                frames = []
                print(f"đã lưu, ấn space để quay tiếp.")
            recording = not recording
        elif key == ord("q"):
            break
# đóng camera
cap.release()
cv2.destroyAllWindows()