"""
PHÂN TÍCH LỖI: accuracy thấp là do đâu? Model nhầm gốc hợp âm (A, B, C...) hay nhầm loại
hợp âm (1, D, M, m1, R)? Class nào kém? Cặp nào hay bị nhầm với nhau?

Chạy: python analyze_errors.py   (dùng cache; KNN với K cố định nên chỉ mất vài chục giây)
"""
from collections import Counter

import numpy as np
import pandas as pd

import common
import knn

K = 5

X_train, X_test, y_train, y_test = common.split_data()
model = knn.KNNScratch(K).fit(X_train[::knn.STRIDE_FIT], y_train[::knn.STRIDE_FIT])
print(f"\nDự đoán {len(X_test)} mẫu test (K={K})...")
pred = model.predict(X_test)

print("\n=== 1. ĐỘ CHÍNH XÁC ===")
print(f"Toàn bộ nhãn (vd C_R) : {common.accuracy(y_test, pred):.4f}")
root_t = pd.Series(y_test).str.split("_").str[0].values
root_p = pd.Series(pred).str.split("_").str[0].values
qual_t = pd.Series(y_test).str.split("_", n=1).str[-1].values
qual_p = pd.Series(pred).str.split("_", n=1).str[-1].values
print(f"Chỉ đoán đúng GỐC      : {common.accuracy(root_t, root_p):.4f}   ({len(set(root_t))} gốc, vd C, Cm, D...)")
print(f"Chỉ đoán đúng LOẠI     : {common.accuracy(qual_t, qual_p):.4f}   ({len(set(qual_t))} loại, vd R, M, 1...)")

print("\n=== 2. THEO SỐ TAY NHẬN ĐƯỢC ===")
n_hands = X_test[:, common.FLAG_COLS].sum(axis=1).astype(int)
for n in (1, 2):
    m = n_hands == n
    if m.any():
        print(f"{n} tay: {m.sum():6d} frame, accuracy {common.accuracy(y_test[m], pred[m]):.4f}")

print("\n=== 3. TỪNG CLASS ===")
tr_count = Counter(y_train)
rows = []
for c in sorted(set(y_test)):
    m = y_test == c
    rows.append((c, tr_count.get(c, 0), int(m.sum()), float((pred[m] == c).mean())))
df = pd.DataFrame(rows, columns=["class", "frame_train", "frame_test", "recall"])
print("10 class kém nhất:")
print(df.sort_values("recall").head(10).to_string(index=False))
print("\n5 class tốt nhất:")
print(df.sort_values("recall", ascending=False).head(5).to_string(index=False))
few = df[df["frame_train"] < 200]
if len(few):
    print(f"\nClass có ít hơn 200 frame train ({len(few)} class): {few['class'].tolist()}")
no_test = sorted(set(y_train) - set(y_test))
if no_test:
    print(f"Class KHÔNG có frame nào ở tập test ({len(no_test)}): {no_test}")

print("\n=== 4. 10 CẶP NHẦM LẪN NHIỀU NHẤT (thật -> đoán) ===")
wrong = Counter((t, p) for t, p in zip(y_test, pred) if t != p)
for (t, p), n in wrong.most_common(10):
    print(f"  {t:8s} -> {p:8s} {n:6d} frame")

print("\n=== 5. THÔNG TIN VỀ GỐC / LOẠI NẰM Ở ĐÂU? (KNN chỉ dùng một nhóm đặc trưng) ===")
groups = {
    "chỉ tay TRÁI (hình dạng)":  list(range(0, 63)) + [126],
    "chỉ tay PHẢI (hình dạng)":  list(range(63, 126)) + [127],
    "cả hai tay (hình dạng)":    list(range(0, 128)),
}
if X_train.shape[1] > 128:
    groups["chỉ VỊ TRÍ cổ tay"] = list(range(128, X_train.shape[1]))
    groups["hình dạng + vị trí"] = list(range(0, X_train.shape[1]))
sub = slice(None, None, 2)          # dùng 1/2 tập test cho nhanh
print(f"{'nhóm đặc trưng':28s} {'gốc':>7s} {'loại':>7s} {'nhãn đủ':>8s}")
for name, cols in groups.items():
    m = knn.KNNScratch(K).fit(X_train[::knn.STRIDE_FIT][:, cols], y_train[::knn.STRIDE_FIT])
    p = m.predict(X_test[sub][:, cols])
    yt = y_test[sub]
    r_t = pd.Series(yt).str.split("_").str[0].values
    r_p = pd.Series(p).str.split("_").str[0].values
    q_t = pd.Series(yt).str.split("_", n=1).str[-1].values
    q_p = pd.Series(p).str.split("_", n=1).str[-1].values
    print(f"{name:28s} {common.accuracy(r_t, r_p):7.3f} {common.accuracy(q_t, q_p):7.3f} "
          f"{common.accuracy(yt, p):8.3f}")