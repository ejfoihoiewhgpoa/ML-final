"""
KIỂM TRA ĐẶC TRƯNG: vì sao accuracy thấp? Xem cột nào thừa, dữ liệu đã chuẩn hóa chưa,
có khung hình rỗng không, là một tay hay hai tay.
Chạy: python check_features.py   (dùng cache của common.py nên chạy nhanh)
"""
import os

import numpy as np
import pandas as pd

import common

# ---------- 1. Nhìn vào MỘT file parquet ----------
meta = pd.read_csv(common.META_CSV)
root = common.DATA_ROOT or os.path.dirname(common.META_CSV)
path = common._resolve(meta[common.PATH_COL].iloc[0], root)
df = pd.read_parquet(path)
print("=== 1. MỘT FILE PARQUET MẪU ===")
print("File :", path)
print("Shape:", df.shape, "(số frame, số cột)")
cols = df.columns.tolist()
print("Cột  :", cols if len(cols) <= 20 else f"{cols[:10]} ... {cols[-8:]}")
print("Kiểu :", dict(df.dtypes.astype(str).value_counts()))
print(df.iloc[0].head(8).to_string())

# ---------- 2. Thống kê toàn bộ ma trận đặc trưng ----------
X, y, g = common.load_data(return_groups=True)
print("\n=== 2. MA TRẬN ĐẶC TRƯNG ===")
print("Shape:", X.shape)
print(f"NaN: {np.isnan(X).sum()} | inf: {np.isinf(X).sum()}")
print(f"Giá trị: min {X.min():.3f} | max {X.max():.3f} | trung bình {X.mean():.3f}")
print(f"Frame toàn số 0 (không thấy tay?): {(np.abs(X).sum(axis=1) == 0).mean():.1%}")

sd, mn, mx = X.std(axis=0), X.min(axis=0), X.max(axis=0)
print(f"Cột hằng số (std=0): {int((sd == 0).sum())}")
big = np.where(sd > 5 * np.median(sd))[0]
print(f"Cột nghi lấn át khoảng cách (std > 5 lần trung vị): {big[:20].tolist()}"
      + (" ..." if len(big) > 20 else ""))
print("Vài cột đầu / cuối:")
for j in [0, 1, 2, X.shape[1] - 2, X.shape[1] - 1]:
    print(f"  cột {j:3d}: min {mn[j]:9.3f} | max {mx[j]:9.3f} | std {sd[j]:8.3f}")

# nếu có 2 khối bằng nhau (2 tay), kiểm tra từng khối có hay bị toàn 0 không
if X.shape[1] >= 100:
    half = X.shape[1] // 2
    for name, blk in [("khối đầu", X[:, :half]), ("khối sau", X[:, half:])]:
        print(f"  {name} ({blk.shape[1]} cột): toàn 0 ở {(np.abs(blk).sum(axis=1) == 0).mean():.1%} số frame")

# ---------- 3. Nhãn ----------
vc = pd.Series(y).value_counts()
print("\n=== 3. NHÃN ===")
print(f"{len(vc)} class | số frame mỗi class: ít nhất {vc.min()}, nhiều nhất {vc.max()}")
print(f"Đoán bừa đúng ~{1 / len(vc):.3f} | luôn đoán class đông nhất đúng {vc.max() / len(y):.3f}")