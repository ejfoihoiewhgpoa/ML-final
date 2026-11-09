"""
CHẨN ĐOÁN ĐƯỜNG DẪN: so sánh cột `path` trong train.csv với các file parquet thật trên ổ đĩa.
Chạy: python check_paths.py   (không đọc nội dung parquet nên chạy nhanh)
"""
import os
from collections import Counter

import pandas as pd

import common

meta = pd.read_csv(common.META_CSV)
root = common.DATA_ROOT or os.path.dirname(common.META_CSV)
print(f"train.csv: {len(meta)} dòng, {meta[common.LABEL_COL].nunique()} label")
print("5 dòng path đầu:", list(meta[common.PATH_COL].head()))

common._resolve("x/x.parquet", root)           # chỉ để kích hoạt quét thư mục
idx, idx_name = common._INDEX, common._INDEX_NAME

# --- phân loại từng dòng của csv ---
kind = []
for p in meta[common.PATH_COL].astype(str):
    p2 = p.replace("\\", "/")
    parts = p2.split("/")
    fn, key = parts[-1], (parts[-2] if len(parts) > 1 else "", parts[-1])
    if os.path.exists(p) or os.path.exists(os.path.join(root, p)):
        kind.append("khớp đường dẫn trực tiếp")
    elif key in idx:
        kind.append("khớp theo (folder cha, tên file)")
    elif len(idx_name.get(fn, [])) == 1:
        kind.append("khớp theo tên file duy nhất")
    elif fn in idx_name:
        kind.append("tên file có trên ổ đĩa nhưng nằm ở folder khác & trùng nhiều chỗ")
    else:
        kind.append("KHÔNG có file nào trên ổ đĩa")
meta["kind"] = kind
print("\n=== KẾT QUẢ KHỚP ===")
print(meta["kind"].value_counts().to_string())

bad = meta[meta["kind"].str.startswith(("KHÔNG", "tên file"))]
if len(bad):
    print("\n=== LABEL THIẾU NHIỀU NHẤT (thiếu / tổng) ===")
    tot = meta[common.LABEL_COL].value_counts()
    for lab, n in bad[common.LABEL_COL].value_counts().head(10).items():
        print(f"  {lab:8s} {n}/{tot[lab]}")
    print("\n=== 3 FILE THIẾU ĐẦU TIÊN: file cùng tên đang nằm ở đâu? ===")
    for p in bad[common.PATH_COL].head(3):
        fn = str(p).replace("\\", "/").split("/")[-1]
        print(f"  csv ghi: {p}")
        for q in idx_name.get(fn, [])[:4]:
            print(f"      trên ổ đĩa có: {q}")
        if fn not in idx_name:
            print("      trên ổ đĩa: không có file nào tên này")

print("\n=== FOLDER TRONG CSV (phần đầu của path) ===")
print(Counter(str(p).replace("\\", "/").split("/")[0] for p in meta[common.PATH_COL]).most_common(15))
print("=== FOLDER CHỨA PARQUET TRÊN Ổ ĐĨA (tên folder cuối, số file) ===")
print(Counter(k[0] for k in idx).most_common(15))