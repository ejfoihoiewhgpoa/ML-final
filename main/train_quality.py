"""
TRAIN MODEL CHO APP: nhận LOẠI hợp âm (1, D, M, m1, R) chỉ từ TAY TRÁI.

Vì sao chỉ tay trái: phân tích trước đó cho thấy hình dạng tay trái đủ để đoán loại hợp âm
(~90%), còn gốc hợp âm không đoán được từ cử chỉ nên app cho chọn gốc bằng phím A-G.

Script này:
  1) Lấy dữ liệu từ common.split_data() (cùng cách chia train/test theo sequence như các file khác)
  2) Chỉ giữ 63 cột hình dạng tay trái + cờ "tay trái có mặt", chỉ dùng frame có tay trái
  3) Train KNN và Random Forest (tự code), đo accuracy, balanced accuracy và thời gian dự đoán
     từng frame (app dự đoán từng frame một nên thời gian này quan trọng)
  4) Chọn model tốt hơn, lưu vào quality_model.pkl (cạnh file này) để app.py nạp

Chạy: python train_quality.py
"""
import pickle
import time
from pathlib import Path

import numpy as np

import common
import knn
from common import accuracy, confusion_matrix, split_data
from decision_tree import LEFT_COLS, balanced_accuracy, convert_labels, select_features
from random_forest import RandomForestScratch

OUT_PATH = Path(__file__).resolve().parent / "quality_model.pkl"
MAX_MS_PER_FRAME = 40      # app cần chạy được ~25 FPS, nên loại model dự đoán chậm hơn mức này


def frame_time_ms(model, X, n=100):
    """Thời gian dự đoán TRUNG BÌNH khi đưa từng frame một (giống lúc chạy webcam)."""
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    t0 = time.time()
    for i in idx:
        model.predict(X[i:i + 1])
    return (time.time() - t0) / len(idx) * 1000


def main():
    X_train, X_test, y_train, y_test = split_data()[:4]
    groups = common.get_train_groups()

    y_train, y_test = convert_labels(y_train, "quality"), convert_labels(y_test, "quality")
    X_train, _ = select_features(X_train, "left")
    X_test, _ = select_features(X_test, "left")

    # chỉ giữ frame mà tay trái được nhận ra (cột cuối là cờ có mặt)
    m_tr, m_te = X_train[:, -1] > 0, X_test[:, -1] > 0
    X_train, y_train = X_train[m_tr], y_train[m_tr]
    X_test, y_test = X_test[m_te], y_test[m_te]
    if groups is not None and len(groups) == len(m_tr):
        common._TRAIN_GROUPS = groups[m_tr]            # giữ nhóm khớp để CV theo sequence vẫn đúng
    classes = sorted(str(c) for c in set(y_train))
    print(f"\nLoại hợp âm: {classes} | train={len(y_train)} test={len(y_test)} frame có tay trái")
    print("Số frame train mỗi loại:", {c: int((y_train == c).sum()) for c in classes})

    candidates = []

    print("\n--- KNN ---")
    k = knn.choose_k(X_train, y_train)
    model = knn.KNNScratch(k).fit(X_train[::knn.STRIDE_FIT], y_train[::knn.STRIDE_FIT])
    pred = model.predict(X_test)
    candidates.append(("KNN", model, pred))

    print("\n--- RANDOM FOREST ---")
    t0 = time.time()
    model = RandomForestScratch(n_estimators=100, max_depth=14).fit(X_train, y_train)
    print(f"  train xong ({time.time() - t0:.0f}s)")
    pred = model.predict(X_test)
    candidates.append(("Random Forest", model, pred))

    print("\n=== KẾT QUẢ (tập test, chỉ frame có tay trái) ===")
    print(f"{'Model':15s} {'accuracy':>9s} {'balanced':>9s} {'ms/frame':>9s}")
    rows = []
    for name, model, pred in candidates:
        ms = frame_time_ms(model, X_test)
        row = dict(name=name, model=model, pred=pred, acc=accuracy(y_test, pred),
                   bal=balanced_accuracy(y_test, pred), ms=ms)
        rows.append(row)
        print(f"{name:15s} {row['acc']:9.4f} {row['bal']:9.4f} {ms:9.1f}")

    ok = [r for r in rows if r["ms"] <= MAX_MS_PER_FRAME] or rows
    best = max(ok, key=lambda r: r["bal"])
    print(f"\n=> Chọn: {best['name']} (balanced accuracy {best['bal']:.3f}, {best['ms']:.1f} ms/frame)")

    cm = confusion_matrix(y_test, best["pred"], classes)
    print("\nMa trận nhầm lẫn (hàng = thật, cột = đoán):")
    print(" " * 6 + "".join(f"{c:>7s}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:>6s}" + "".join(f"{v:7d}" for v in row) + f"   recall {row[classes.index(c)] / max(row.sum(), 1):.2f}")

    info = {"model": best["model"], "cols": LEFT_COLS, "name": best["name"], "classes": classes,
            "metrics": {"accuracy": best["acc"], "balanced_accuracy": best["bal"], "ms_per_frame": best["ms"]}}
    with open(OUT_PATH, "wb") as f:
        pickle.dump(info, f)
    print(f"\nĐã lưu {OUT_PATH}. Bây giờ chạy: python app.py")


if __name__ == "__main__":
    main()