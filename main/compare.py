"""
COMPARE: chạy 3 model tự code trên CÙNG một tập train/test, so sánh kết quả,
vẽ confusion matrix và lưu model.

Chạy: python compare.py
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np

import decision_tree
from decision_tree import balanced_accuracy, feature_name
import knn
import random_forest
from common import (accuracy, classification_report, confusion_matrix,
                    split_data)

X_train, X_test, y_train, y_test = split_data()

knn_model, pred_knn = knn.run(X_train, X_test, y_train, y_test)
dt_model, pred_dt = decision_tree.run(X_train, X_test, y_train, y_test)
rf_model, pred_rf = random_forest.run(X_train, X_test, y_train, y_test)

results = {
    "KNN": (knn_model, pred_knn),
    "Decision Tree": (dt_model, pred_dt),
    "Random Forest": (rf_model, pred_rf),
}

# ---------------- So sánh ----------------
print("\n=== SO SÁNH TRÊN TẬP TEST ===")
print(f"  {'Model':15s} {'accuracy':>9s} {'balanced':>9s}")
for name, (_, pred) in results.items():
    print(f"  {name:15s} {accuracy(y_test, pred):9.4f} {balanced_accuracy(y_test, pred):9.4f}")

best_name = max(results, key=lambda n: accuracy(y_test, results[n][1]))
classes = list(np.unique(np.concatenate([y_train, y_test])))
print(f"\nModel tốt nhất: {best_name}")
classification_report(y_test, results[best_name][1], classes)

# ---------------- Confusion matrix ----------------
fig, axes = plt.subplots(1, 3, figsize=(30, 10))
for ax, (name, (_, pred)) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, pred, classes)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"{name} (acc={accuracy(y_test, pred):.3f})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix_3_models.png", dpi=100)
print("Đã lưu confusion_matrix_3_models.png")

# ---------------- Đặc trưng quan trọng (Random Forest) ----------------
imp = rf_model.feature_importances_
print("\nTop 5 đặc trưng quan trọng nhất (Random Forest):")
for i in np.argsort(imp)[::-1][:5]:
    print(f"  {feature_name(i):32s} (độ quan trọng {imp[i]:.3f})")

# ---------------- Lưu model (pickle, không cần thư viện ngoài) ----------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(results[best_name][0], f)
with open("three_models.pkl", "wb") as f:
    pickle.dump({"KNN": knn_model, "DT": dt_model, "RF": rf_model}, f)
print(f"\nĐã lưu best_model.pkl ({best_name}) và three_models.pkl")

# ---------------- Lưu dự đoán + vẽ thêm các hình (plot_results.py) ----------------
np.savez("predictions.npz", y_test=y_test, KNN=pred_knn, DT=pred_dt, RF=pred_rf)
try:
    import plot_results
    plot_results.make_all({name: model for name, (model, _) in results.items()},
                          y_test, {name: pred for name, (_, pred) in results.items()})
except Exception as e:                  # vẽ lỗi cũng không ảnh hưởng phần train / lưu model
    print(f"[Bỏ qua phần vẽ thêm] {type(e).__name__}: {e}")

# Dùng khi chạy webcam (nhớ để các file .py này cùng thư mục để pickle load được):
#   import pickle
#   from common import normalize
#   model = pickle.load(open("best_model.pkl", "rb"))
#   feat = normalize(res.multi_hand_landmarks[0].landmark)
#   label = model.predict(feat)[0]     # ví dụ "A_m1" -> tra bảng LABEL_TO_SOUND