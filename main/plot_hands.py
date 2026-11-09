"""
VẼ BÀN TAY TRUNG BÌNH: tay nào thay đổi khi đổi gốc hợp âm? tay nào thay đổi khi đổi loại?
Mỗi ô là bộ xương (21 điểm) trung bình của một nhóm frame, đã chuẩn hóa.

Tạo 2 ảnh:
  hands_by_root.png    : hàng trên = tay PHẢI theo từng gốc, hàng dưới = tay TRÁI theo từng gốc
  hands_by_quality.png : hàng trên = tay TRÁI theo từng loại, hàng dưới = tay PHẢI theo từng loại
Nếu một tay là nơi thể hiện gốc, thì các ô của tay đó ở ảnh 1 phải KHÁC nhau rõ rệt giữa các gốc.

Chạy: python plot_hands.py   (dùng cache của common.py nên chạy nhanh)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
         (18, 19), (19, 20), (0, 17)]
FINGER_COLORS = ["tab:red"] * 4 + ["tab:orange"] * 4 + ["tab:green"] * 4 + \
                ["tab:blue"] * 4 + ["tab:purple"] * 4

X, y = common.load_data()
roots = pd.Series(y).str.split("_").str[0].values
quals = pd.Series(y).str.split("_", n=1).str[-1].values
LEFT, RIGHT = slice(0, 63), slice(63, 126)
PRESENT = {"TRÁI": 126, "PHẢI": 127}


def mean_hand(mask, block, flag_col):
    m = mask & (X[:, flag_col] > 0)            # chỉ frame mà tay đó được nhận ra
    if m.sum() == 0:
        return None, 0
    return X[m][:, block].mean(axis=0).reshape(21, 3), int(m.sum())


def draw(ax, pts, title):
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    if pts is None:
        return
    x, yy = pts[:, 0], -pts[:, 1]               # đảo trục y: ảnh có y hướng xuống
    for (a, b) in EDGES:
        ax.plot([x[a], x[b]], [yy[a], yy[b]], color="gray", lw=1)
    for i in range(21):
        ax.scatter(x[i], yy[i], s=8, color=FINGER_COLORS[min(i - 1, 19)] if i else "black", zorder=3)


def figure(groups, labels_arr, rows, fname, suptitle):
    ncol = len(groups)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(1.9 * ncol, 2.6 * len(rows)), squeeze=False)
    for r, (hand_name, block, flag) in enumerate(rows):
        for c, g in enumerate(groups):
            pts, n = mean_hand(labels_arr == g, block, flag)
            draw(axes[r][c], pts, f"{hand_name}: {g}\n(n={n})")
    fig.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(fname, dpi=110)
    plt.close(fig)
    print("Đã lưu", fname)


root_list = sorted(set(roots))
qual_list = sorted(set(quals))
figure(root_list, roots,
       [("tay PHẢI", RIGHT, PRESENT["PHẢI"]), ("tay TRÁI", LEFT, PRESENT["TRÁI"])],
       "hands_by_root.png", "Bàn tay trung bình theo GỐC hợp âm (gộp mọi loại)")
figure(qual_list, quals,
       [("tay TRÁI", LEFT, PRESENT["TRÁI"]), ("tay PHẢI", RIGHT, PRESENT["PHẢI"])],
       "hands_by_quality.png", "Bàn tay trung bình theo LOẠI hợp âm (gộp mọi gốc)")