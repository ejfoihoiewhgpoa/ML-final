"""
Vẽ thống kê cho Decision Tree, cùng phong cách plot_hands.py.

LAYOUT CỘT (theo plot_hands.py / common.py):
  X[:, 0:63]    tay TRÁI  = 21 điểm x (x, y, z)
  X[:, 63:126]  tay PHẢI  = 21 điểm x (x, y, z)
  X[:, 126]     cờ: tay TRÁI được nhận ra (>0)
  X[:, 127]     cờ: tay PHẢI được nhận ra (>0)
  X[:, 128:137] (nếu EXTRA_POSITION) vị trí tuyệt đối: cổ tay trái xyz, cổ tay phải xyz, phải - trái

Tạo (lưu cạnh file này):
  dt_confusion.png       confusion matrix (chuẩn hóa theo hàng) + recall từng lớp
  dt_hand_importance.png bàn tay trái / phải, landmark tô màu theo độ quan trọng với cây
  dt_top_features.png    top-N đặc trưng quan trọng nhất
  dt_mean_vs_pred.png    bàn tay trung bình theo nhãn DỰ ĐOÁN (so với hands_by_quality.png)
  dt_rules.txt           luật if/else vài tầng đầu

Dùng:
    from plot_decision_tree import plot_all
    plot_all(model, X_train, X_test, y_test, pred)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # chỉ lưu file, không cần mở cửa sổ
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

N_HAND = 63                    # 21 điểm x 3
LEFT, RIGHT = slice(0, 63), slice(63, 126)
PRESENT = {"TRÁI": 126, "PHẢI": 127}
HAND_NAMES = ["TRÁI", "PHẢI"]

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
         (18, 19), (19, 20), (0, 17)]
FINGER_COLORS = ["tab:red"] * 4 + ["tab:orange"] * 4 + ["tab:green"] * 4 + \
                ["tab:blue"] * 4 + ["tab:purple"] * 4
FINGER_NAMES = ["cái", "trỏ", "giữa", "áp út", "út"]


# ----------------------------------------------------------------------
def get_importances(model):
    """Cây: importances_ ; Rừng: feature_importances_."""
    return model.importances_ if hasattr(model, "importances_") else model.feature_importances_


def hands_of(X):
    """(n, F>=126) -> (n, 2, 21, 3): [tay trái, tay phải]."""
    X = np.atleast_2d(X)
    if X.shape[1] < 126:
        raise ValueError(f"X chỉ có {X.shape[1]} cột, cần >= 126 (2 tay x 21 điểm x 3).")
    return X[:, :126].reshape(len(X), 2, 21, 3)


def has_flags(X):
    return X.shape[1] >= 128


POS_NAMES = ["cổ tay TRÁI", "cổ tay PHẢI", "(phải − trái)"]


def feature_name(j):
    if j == 126 or j == 127:
        return f"cờ nhận ra tay {'TRÁI' if j == 126 else 'PHẢI'}"
    if j >= 128:                                   # đặc trưng VỊ TRÍ tuyệt đối
        k = j - 128
        return f"VỊ TRÍ {POS_NAMES[min(k // 3, 2)]} · {'xyz'[k % 3]}"
    hand, rest = divmod(j, N_HAND)
    lm, ax = divmod(rest, 3)
    finger = "cổ tay" if lm == 0 else f"ngón {FINGER_NAMES[(lm - 1) // 4]}"
    return f"{HAND_NAMES[hand]} · {finger} · điểm {lm} · {'xyz'[ax]}"


def mean_hands(X, mask=None):
    """Bàn tay trung bình (2, 21, 3) + số frame mỗi tay; chỉ tính frame mà tay được nhận ra."""
    X = np.asarray(X, float)
    mask = np.ones(len(X), bool) if mask is None else mask
    H = hands_of(X)
    out, ns = np.full((2, 21, 3), np.nan), [0, 0]
    for h, name in enumerate(HAND_NAMES):
        m = mask & (X[:, PRESENT[name]] > 0) if has_flags(X) else mask
        ns[h] = int(m.sum())
        if m.any():
            out[h] = H[m, h].mean(0)
    return out, ns


def draw_hand(ax, pts, imp=None, vmax=None):
    """pts: (21, 3). Lật trục y như plot_hands.py (ảnh có y hướng xuống)."""
    x, y = pts[:, 0], -pts[:, 1]
    for a, b in EDGES:
        ax.plot([x[a], x[b]], [y[a], y[b]], color="gray", lw=1, zorder=1)
    if imp is None:
        for i in range(21):
            ax.scatter(x[i], y[i], s=10, color=FINGER_COLORS[i - 1] if i else "black", zorder=3)
        return None
    size = 15 + 500 * (imp / (vmax + 1e-12))
    return ax.scatter(x, y, c=imp, s=size, cmap="YlOrRd", vmin=0, vmax=vmax,
                      edgecolors="k", linewidths=0.4, zorder=3)


def _style(ax, title, lim=None):
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if lim is not None:
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    else:
        ax.margins(0.15)


def _limits(hands_list, pad=0.4):
    pts = np.concatenate([h[:, :2].reshape(-1, 2) * [1, -1] for h in hands_list])
    pts = pts[~np.isnan(pts).any(1)]
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    return lo[0], hi[0], lo[1], hi[1]


# ----------------------------------------------------------------------
def confusion(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)))
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


def plot_confusion(y_true, y_pred, classes, path, name=""):
    M = confusion(y_true, y_pred, classes)
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    recall = np.diag(Mn)
    k = len(classes)
    big = k > 12

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(11, 0.5 * k + 7), max(4.6, 0.35 * k + 3)),
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    im = a1.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
    a1.set_xticks(range(k)); a1.set_xticklabels(classes, rotation=90 if big else 0, fontsize=7 if big else 9)
    a1.set_yticks(range(k)); a1.set_yticklabels(classes, fontsize=7 if big else 9)
    a1.set_xlabel("Dự đoán"); a1.set_ylabel("Thực tế")
    a1.set_title(f"Confusion matrix {name} (chuẩn hóa theo hàng)")
    if not big:
        for i in range(k):
            for j in range(k):
                a1.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center",
                        color="white" if Mn[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=a1, fraction=0.046)

    a2.bar(range(k), recall, color="tab:blue")
    a2.set_xticks(range(k)); a2.set_xticklabels(classes, rotation=90 if big else 0, fontsize=7 if big else 9)
    a2.axhline(1 / k, color="r", ls="--", lw=1, label="đoán bừa")
    a2.axhline(recall.mean(), color="k", ls=":", lw=1, label=f"balanced acc = {recall.mean():.3f}")
    a2.set_ylim(0, 1); a2.set_title("Recall từng lớp"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_hand_importance(model, X_ref, path, name="Decision Tree"):
    imp_all = get_importances(model)
    imp = imp_all[:126].reshape(2, 21, 3).sum(-1)            # (2, 21)
    flag_imp = imp_all[126:128].sum() if len(imp_all) > 126 else 0.0
    pos_imp = imp_all[128:].sum() if len(imp_all) > 128 else 0.0
    mh, _ = mean_hands(X_ref)
    vmax = imp.max() if imp.max() > 0 else 1.0
    lim = _limits([mh[0], mh[1]])

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    sc = None
    for h in range(2):
        pts = np.nan_to_num(mh[h])
        sc = draw_hand(axes[h], pts, imp[h], vmax)
        _style(axes[h], f"tay {HAND_NAMES[h]}  (tổng importance = {imp[h].sum():.2f})", lim)
    fig.colorbar(sc, ax=axes, fraction=0.03, label="độ quan trọng của landmark")
    extra = f"   [hình dạng tay: {imp.sum():.2f} | vị trí cổ tay: {pos_imp:.2f} | cờ: {flag_imp:.2f}]"
    fig.suptitle(f"Landmark nào được {name} dùng?" + extra, fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_top_features(model, top, path, name=""):
    imp = get_importances(model)
    order = np.argsort(imp)[::-1][:top]
    names = [feature_name(j) for j in order][::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(names, imp[order][::-1], color="tab:orange")
    ax.set_title(f"Top {top} đặc trưng quan trọng nhất {name}"); ax.set_xlabel("importance")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_mean_by_pred(X_test, y_pred, classes, path, max_cols=8, name="cây"):
    """Giống hands_by_quality.png nhưng gộp theo nhãn DỰ ĐOÁN của cây (tối đa max_cols lớp/ảnh)."""
    X_test, y_pred = np.asarray(X_test, float), np.asarray(y_pred)
    classes = list(classes)
    chunks = [classes[i:i + max_cols] for i in range(0, len(classes), max_cols)]
    for ci, chunk in enumerate(chunks):
        means = [mean_hands(X_test, y_pred == c) for c in chunk]
        lim = _limits([m[0][h] for m in means for h in range(2)])
        fig, axes = plt.subplots(2, len(chunk), figsize=(1.9 * len(chunk), 5.2), squeeze=False)
        for j, (c, (mh, ns)) in enumerate(zip(chunk, means)):
            for h in range(2):
                ax = axes[h][j]
                if ns[h] > 0:
                    draw_hand(ax, mh[h])
                _style(ax, f"tay {HAND_NAMES[h]}: {c}\n(n={ns[h]})", lim)
        fig.suptitle(f"Bàn tay trung bình theo nhãn {name} DỰ ĐOÁN")
        fig.tight_layout()
        fp = path if len(chunks) == 1 else path.replace(".png", f"_{ci + 1}.png")
        fig.savefig(fp, dpi=110); plt.close(fig)


def export_rules(model, max_depth=3):
    lines = []

    def rec(node, depth):
        pad = "|   " * depth
        if node.is_leaf() or depth > max_depth:
            lines.append(f"{pad}=> {model.classes_[node.value]}")
            return
        name = feature_name(node.feature)
        lines.append(f"{pad}nếu {name} <= {node.threshold:.4f}:")
        rec(node.left, depth + 1)
        lines.append(f"{pad}ngược lại ({name} > {node.threshold:.4f}):")
        rec(node.right, depth + 1)

    rec(model.root, 0)
    return "\n".join(lines)


def plot_all(model, X_train, X_test, y_test, y_pred, top=15, out_dir=None,
             prefix="dt", name="Decision Tree"):
    """
    Lưu hình vào out_dir (mặc định: cùng thư mục với file này).
    prefix: tiền tố tên file ("dt" cho cây, "rf" cho rừng) để không đè lên nhau.
    Dùng được cho cả DecisionTreeScratch và RandomForestScratch.
    """
    out_dir = out_dir or SCRIPT_DIR
    os.makedirs(out_dir, exist_ok=True)
    P = lambda fn: os.path.join(out_dir, f"{prefix}_{fn}")
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    classes = list(model.classes_)

    plot_confusion(y_test, y_pred, classes, P("confusion.png"), name)
    plot_hand_importance(model, X_train, P("hand_importance.png"), name)
    plot_top_features(model, top, P("top_features.png"), name)
    plot_mean_by_pred(X_test, y_pred, classes, P("mean_vs_pred.png"), name=name)
    if hasattr(model, "root"):                      # chỉ cây đơn mới có luật if/else để in
        with open(P("rules.txt"), "w", encoding="utf-8") as f:
            f.write(export_rules(model))

    print(f"\nĐã lưu ảnh PNG ({name}) vào thư mục:\n  {out_dir}")
    for fn in sorted(os.listdir(out_dir)):
        if fn.startswith(prefix + "_") and fn.endswith((".png", ".txt")):
            print("  -", os.path.join(out_dir, fn))