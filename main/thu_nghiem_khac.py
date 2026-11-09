"""
THỬ NGHIỆM: so sánh nhiều model ML khác (file này ĐƯỢC dùng sklearn, vì chỉ để thử thêm).
Các dòng [sklearn] KNN / Decision Tree / Random Forest là bản thư viện của 3 model chính,
để đối chiếu với bản tự code.

Dùng chung split_data() nên cùng tập train/test với compare.py.

Vì dữ liệu lớn (~220k frame train, 56 lớp), file này chạy trên MỘT PHẦN dữ liệu:
  - STRIDE_TRAIN: chỉ lấy 1 trong mỗi STRIDE_TRAIN frame của tập train
  - STRIDE_TEST : tương tự cho tập test
  - SVM (RBF) chậm theo bình phương số mẫu nên chỉ train trên tối đa SVM_MAX mẫu
Các frame liền kề gần như giống nhau nên lấy bớt gần như không mất thông tin.
Muốn chạy đầy đủ thì đặt STRIDE = 1 (sẽ lâu hơn nhiều).

Mỗi model được bọc try/except: model nào lỗi hoặc quá chậm thì bạn bỏ nó khỏi
build_models() là được, các model khác vẫn chạy.

Chạy: python thu_nghiem_khac.py
"""
import time

import numpy as np
import pandas as pd

from common import SEED, accuracy, split_data
from decision_tree import balanced_accuracy, convert_labels, select_features

LABEL = "full"          # "full" (gốc_loại) | "root" | "quality"
FEATURES = "all"        # "all" | "left" | "right"
STRIDE_TRAIN = 5
STRIDE_TEST = 2
SVM_MAX = 15000


def build_models():
    """Trả về dict {tên: (hàm tạo model, số mẫu train tối đa hoặc None)}."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, LinearSVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        # --- bản sklearn của 3 model chính (để đối chiếu với bản tự code) ---
        "[sklearn] KNN":           (lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=-1), None),
        "[sklearn] Decision Tree": (lambda: DecisionTreeClassifier(max_depth=14, min_samples_leaf=20,
                                                                   class_weight="balanced", random_state=SEED), None),
        "[sklearn] Random Forest": (lambda: RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=20,
                                                                   class_weight="balanced", n_jobs=-1,
                                                                   random_state=SEED), None),
        # --- thử thêm ---
        "KNN + chuẩn hóa":         (lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, n_jobs=-1)), None),
        "Logistic Regression":     (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced")), None),
        "SVM tuyến tính":          (lambda: make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", dual=False)), None),
        "SVM (RBF)":               (lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, class_weight="balanced")), SVM_MAX),
        "Naive Bayes":             (lambda: GaussianNB(), None),
    }


def main():
    X_train, X_test, y_train, y_test = split_data()[:4]
    y_train, y_test = convert_labels(y_train, LABEL), convert_labels(y_test, LABEL)
    X_train, _ = select_features(X_train, FEATURES)
    X_test, _ = select_features(X_test, FEATURES)
    X_train, y_train = X_train[::STRIDE_TRAIN], y_train[::STRIDE_TRAIN]
    X_test, y_test = X_test[::STRIDE_TEST], y_test[::STRIDE_TEST]
    print(f"Nhãn: {LABEL} | đặc trưng: {FEATURES} | train={len(y_train)} test={len(y_test)} "
          f"| {len(set(y_train))} lớp\n")

    rows = []
    for name, (make, max_train) in build_models().items():
        Xt, yt = (X_train, y_train) if max_train is None else (X_train[:max_train], y_train[:max_train])
        try:
            t0 = time.time()
            model = make().fit(Xt, yt)
            t_fit = time.time() - t0
            t0 = time.time()
            pred = model.predict(X_test)
            t_pred = (time.time() - t0) / len(X_test) * 1000          # ms / mẫu (quan trọng khi realtime)
        except Exception as e:
            print(f"{name:26s} LỖI: {type(e).__name__}: {e}")
            continue
        rows.append({"Model": name, "Accuracy": accuracy(y_test, pred),
                     "Balanced acc": balanced_accuracy(y_test, pred),
                     "Train (s)": t_fit, "Predict (ms/mẫu)": t_pred, "Số mẫu train": len(Xt)})
        print(f"{name:26s} acc={rows[-1]['Accuracy']:.4f}  bal={rows[-1]['Balanced acc']:.4f}  "
              f"train={t_fit:6.1f}s  predict={t_pred:.3f}ms/mẫu")

    res = pd.DataFrame(rows).sort_values("Balanced acc", ascending=False).reset_index(drop=True)
    print("\n===== BẢNG XẾP HẠNG (theo balanced accuracy) =====")
    print(res.round(4).to_string(index=False))
    res.to_csv("so_sanh_model.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        colors = ["tab:orange" if n.startswith("[sklearn]") else "tab:blue" for n in res["Model"]]
        plt.barh(res["Model"][::-1], res["Balanced acc"][::-1], color=colors[::-1])
        plt.xlabel("Balanced accuracy trên tập test")
        plt.title("So sánh các model (cam = bản sklearn của 3 model chính)")
        plt.tight_layout()
        plt.savefig("so_sanh_model.png", dpi=120)
        print("\nĐã lưu so_sanh_model.csv và so_sanh_model.png")
    except Exception as e:
        print(f"[Bỏ qua phần vẽ] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()