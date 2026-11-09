"""
VẼ KẾT QUẢ: các hình để minh họa / báo cáo. Tất cả lưu thành file PNG cạnh file này.

  plot_1_so_sanh_model.png      : accuracy, balanced accuracy, đoán đúng gốc, đoán đúng loại của 3 model
  plot_2_dac_trung.png          : đặc trưng quan trọng nhất của Decision Tree, Random Forest và
                                  tổng độ quan trọng theo nhóm (tay trái / tay phải / vị trí)
  plot_3_recall_tung_class.png  : recall của từng class (model tốt nhất)
  plot_4_confusion.png          : ma trận nhầm lẫn đã chuẩn hóa theo hàng (model tốt nhất)
  plot_5_cay_quyet_dinh.png     : 3 tầng đầu của Decision Tree

Được gọi tự động từ compare.py. Hoặc chạy riêng: python plot_results.py
(cần đã chạy compare.py trước để có three_models.pkl và predictions.npz)
"""
import pickle

import matplotlib
matplotlib.use("Agg")                      # chỉ lưu file, không mở cửa sổ
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import accuracy, confusion_matrix
from decision_tree import balanced_accuracy, feature_name

COLORS = {"KNN": "#4C72B0", "Decision Tree": "#DD8452", "Random Forest": "#55A868"}


def _parts(y):
    s = pd.Series(np.asarray(y).astype(str))
    return s.str.split("_").str[0].values, s.str.split("_", n=1).str[-1].values


# ----------------------------------------------------------------------
def plot_model_comparison(preds, y_test, fname="plot_1_so_sanh_model.png"):
    names = list(preds)
    has_parts = "_" in str(y_test[0])
    metrics = {"Accuracy\n(nhãn đủ)": lambda p: accuracy(y_test, p),
               "Balanced\naccuracy": lambda p: balanced_accuracy(y_test, p)}
    if has_parts:
        rt, qt = _parts(y_test)
        metrics["Chỉ đoán\nđúng GỐC"] = lambda p: accuracy(rt, _parts(p)[0])
        metrics["Chỉ đoán\nđúng LOẠI"] = lambda p: accuracy(qt, _parts(p)[1])

    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [f(preds[name]) for f in metrics.values()]
        xs = np.arange(len(vals)) + (i - (len(names) - 1) / 2) * w
        bars = ax.bar(xs, vals, w, label=name, color=COLORS.get(name))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(list(metrics))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Độ chính xác trên tập test")
    ax.set_title("So sánh 3 model")
    ax.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print("Đã lưu", fname)


# ----------------------------------------------------------------------
def _color_of(i):
    if i < 63:
        return "#4C72B0"           # tay trái
    if i < 126:
        return "#DD8452"           # tay phải
    return "#8C8C8C"               # cờ / vị trí


def plot_feature_importance(models, fname="plot_2_dac_trung.png", top=15):
    panels = []
    if "Decision Tree" in models:
        panels.append(("Decision Tree", models["Decision Tree"].importances_))
    if "Random Forest" in models:
        panels.append(("Random Forest", models["Random Forest"].feature_importances_))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(6 * (len(panels) + 1), 6))
    for ax, (name, imp) in zip(axes, panels):
        order = np.argsort(imp)[::-1][:top][::-1]
        ax.barh([feature_name(i) for i in order], imp[order], color=[_color_of(i) for i in order])
        ax.set_title(f"Top {top} đặc trưng - {name}")
        ax.tick_params(axis="y", labelsize=7)
    # tổng theo nhóm (dùng Random Forest nếu có)
    imp = panels[-1][1]
    groups = {"tay trái\n(hình dạng)": imp[0:63].sum(), "tay phải\n(hình dạng)": imp[63:126].sum(),
              "cờ tay\ncó mặt": imp[126:128].sum()}
    if len(imp) > 128:
        groups["vị trí\ncổ tay"] = imp[128:].sum()
    ax = axes[-1]
    ax.bar(list(groups), list(groups.values()), color=["#4C72B0", "#DD8452", "#8C8C8C", "#8C8C8C"][:len(groups)])
    for i, v in enumerate(groups.values()):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_title(f"Tổng độ quan trọng theo nhóm ({panels[-1][0]})")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print("Đã lưu", fname)


# ----------------------------------------------------------------------
def plot_per_class_recall(y_test, pred, name, fname="plot_3_recall_tung_class.png"):
    y_test, pred = np.asarray(y_test), np.asarray(pred)
    rows = [(c, float(np.mean(pred[y_test == c] == c)), int((y_test == c).sum())) for c in np.unique(y_test)]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(rows))))
    ax.barh([f"{c} (n={n})" for c, _, n in rows], [r for _, r, _ in rows],
            color=["#C44E52" if r < 0.2 else "#55A868" if r >= 0.6 else "#4C72B0" for _, r, _ in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recall (tỉ lệ frame của class đó được đoán đúng)")
    ax.set_title(f"Recall từng class - {name}\n(đỏ < 0.2, xanh lá >= 0.6)")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print("Đã lưu", fname)


# ----------------------------------------------------------------------
def plot_confusion(y_test, pred, name, fname="plot_4_confusion.png"):
    classes = sorted(set(np.asarray(y_test)) | set(np.asarray(pred)))
    cm = confusion_matrix(y_test, pred, classes).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)          # mỗi hàng cộng lại = 1
    size = max(7, 0.28 * len(classes))
    fig, ax = plt.subplots(figsize=(size + 1, size))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(f"Ma trận nhầm lẫn (chuẩn hóa theo hàng) - {name}\nĐường chéo đậm = đoán đúng")
    fig.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print("Đã lưu", fname)


# ----------------------------------------------------------------------
def plot_tree_top(tree, fname="plot_5_cay_quyet_dinh.png", max_depth=3):
    fig, ax = plt.subplots(figsize=(22, 8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)

    def draw(node, depth, x0, x1):
        x, y = (x0 + x1) / 2, 1 - depth / (max_depth + 0.4)
        if node.is_leaf() or depth == max_depth:
            kind = "lá" if node.is_leaf() else "(còn chia tiếp)"
            ax.text(x, y, f"đoán: {tree.classes_[node.value]}\n{kind}", ha="center", va="center",
                    fontsize=8, bbox=dict(boxstyle="round", fc="#CFE8CF", ec="gray"))
            return x, y
        label = feature_name(node.feature).replace(", ", "\n") + f"\n<= {node.threshold:.2f} ?"
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc="#FFE9B5", ec="gray"))
        for child, (a, b), tag in ((node.left, (x0, x), "đúng"), (node.right, (x, x1), "sai")):
            cx, cy = draw(child, depth + 1, a, b)
            ax.annotate("", xy=(cx, cy + 0.07), xytext=(x, y - 0.07),
                        arrowprops=dict(arrowstyle="->", color="gray"))
            ax.text((x + cx) / 2, (y + cy) / 2 + 0.01, tag, fontsize=8, color="gray", ha="center")
        return x, y

    draw(tree.root, 0, 0.0, 1.0)
    ax.set_title(f"Decision Tree - {max_depth} tầng đầu (ô vàng: câu hỏi, ô xanh: kết quả đoán)")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close(fig)
    print("Đã lưu", fname)


# ----------------------------------------------------------------------
def make_all(models, y_test, preds):
    """models, preds: dict {"KNN": ..., "Decision Tree": ..., "Random Forest": ...}."""
    best = max(preds, key=lambda n: accuracy(y_test, preds[n]))
    steps = [
        lambda: plot_model_comparison(preds, y_test),
        lambda: plot_feature_importance(models),
        lambda: plot_per_class_recall(y_test, preds[best], best),
        lambda: plot_confusion(y_test, preds[best], best),
        lambda: plot_tree_top(models["Decision Tree"]) if "Decision Tree" in models else None,
    ]
    for step in steps:                      # một hình lỗi không làm mất các hình còn lại
        try:
            step()
        except Exception as e:
            print(f"  [Bỏ qua một hình] {type(e).__name__}: {e}")


if __name__ == "__main__":
    with open("three_models.pkl", "rb") as f:
        m = pickle.load(f)
    p = np.load("predictions.npz", allow_pickle=False)
    make_all({"KNN": m["KNN"], "Decision Tree": m["DT"], "Random Forest": m["RF"]},
             p["y_test"], {"KNN": p["KNN"], "Decision Tree": p["DT"], "Random Forest": p["RF"]})