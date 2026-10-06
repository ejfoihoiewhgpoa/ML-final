import os
import numpy as np
import pandas as pd
 
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "my_dataset")

# đọc train.csv
CSV_PATH = os.path.join(DATA, "train.csv") 
df = pd.read_csv(CSV_PATH) 
df["label"] = df["chord"]

LABEL_MAP = {
    "none": 0,
    "A": 1, 
    "Am": 2,
    "C": 3,
    "Cm": 4,
    "D": 5,
    "Dm": 6,
    "E": 7,
    "Em": 8,
    "F": 9,
    "G": 10,
}

def label_to_id(name):                            
    return LABEL_MAP[name]

COLS = [f"{a}_{side}_hand_{i}"                    # 63 cột tay trái rồi 63 cột tay phải
        for side in ("left", "right") for i in range(21) for a in "xyz"]

rows = []
for label in sorted(os.listdir(DATA)):
    folder = os.path.join(DATA, label)
    if not os.path.isdir(folder):                 # bỏ qua file lẻ như train.csv
        continue
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".parquet"):
            rows.append({"path": os.path.join(label, fname), "label": label})
df = pd.DataFrame(rows)

def load_clip(path):
    seq = pd.read_parquet(path)
    h = seq[COLS].dropna().to_numpy()             # bỏ frame thiếu bất kỳ tay nào
    if len(h) == 0:
        return np.empty((0, 129))
    h = h.reshape(-1, 2, 21, 3)                   # (frame, 2 tay, 21 điểm, xyz)

    wrist = h[:, :, 0:1, :]                       # cổ tay mỗi tay
    rel = h - wrist                               # tính từ cổ tay
    scale = np.linalg.norm(rel[:, :, 9, :], axis=2)        # cổ tay -> gốc ngón giữa, (frame, 2)
    scale = np.maximum(scale, 1e-6) # tránh chia cho 0
    rel = rel / scale[:, :, None, None]           # chuẩn hóa kích thước từng tay

    # vị trí tương đối giữa hai cổ tay (giữ thông tin phối hợp hai tay)
    offset = (wrist[:, 1, 0, :] - wrist[:, 0, 0, :]) / scale.mean(axis=1, keepdims=True)

    return np.hstack([rel.reshape(len(rel), 126), offset])   # (frame, 129)

# 3. Gom dữ liệu: mỗi frame hợp lệ là một mẫu
X_list, y_list, g_list = [], [], []
for clip_id, row in df.iterrows():
    feats = load_clip(os.path.join(DATA, row["path"]))
    X_list.append(feats)
    y_list += [label_to_id(row["label"])] * len(feats)    # frame kế thừa nhãn của clip
    g_list += [clip_id] * len(feats)                      # frame này thuộc clip nào

X = np.vstack(X_list)
y = np.array(y_list)
groups = np.array(g_list)

# chia train/test theo clip
# một clip chỉ xuất hiện ở train hoặc test
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
train_idx, test_idx = next(gss.split(X, y, groups))
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
print("train:", X_train.shape, "test:", X_test.shape)