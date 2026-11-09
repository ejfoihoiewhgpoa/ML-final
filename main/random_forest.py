"""
RANDOM FOREST (Rừng ngẫu nhiên) - tự code, dùng lại DecisionTreeScratch

Ý tưởng: nhiều Decision Tree "bỏ phiếu" với nhau.
  1) Bootstrap: mỗi cây train trên 1 tập mẫu lấy ngẫu nhiên CÓ HOÀN LẠI từ tập train
  2) Mỗi lần chia, cây chỉ xét 1 tập con ngẫu nhiên các đặc trưng
     (max_features="sqrt" -> ~căn(137) ~ 11 đặc trưng) -> các cây ít giống nhau
  3) Dự đoán: mọi cây bỏ phiếu, label nhiều phiếu nhất thắng
Vì sao tốt hơn 1 cây: mỗi cây sai theo một kiểu khác nhau, bỏ phiếu thì sai số bù trừ.

Để chạy nhanh với ~275k frame:
  - max_samples: mỗi cây chỉ bootstrap tối đa ngần này frame (mặc định 20000). Frame liền kề
    gần như giống nhau nên lấy ít hơn vẫn giữ gần đủ thông tin, mỗi cây nhanh hơn nhiều.
  - Chia khoảng (histogram) của đặc trưng làm MỘT lần cho cả rừng, các cây dùng chung.
  - max_depth=14, min_samples_leaf=20: cây không mọc tới tận từng frame (đỡ overfit, nhanh).
  - class_weight="balanced": class ít frame có trọng số lớn hơn.
  - CV theo nhóm (sequence) trên 1/stride số frame để chọn số cây.

Chạy riêng: python random_forest.py
"""
import numpy as np

from common import SEED, accuracy, get_train_groups, split_data
from decision_tree import (DecisionTreeScratch, balanced_accuracy, bin_data, convert_labels,
                           feature_name, group_cv_score, make_bins, select_features)


class RandomForestScratch:
    def __init__(self, n_estimators=100, max_depth=14, max_features="sqrt",
                 min_samples_leaf=20, max_samples=20000, class_weight="balanced",
                 seed=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_samples = max_samples        # None | số nguyên (số frame) | số thực 0..1 (tỉ lệ)
        self.class_weight = class_weight
        self.seed = seed

    def _bootstrap_size(self, n):
        if self.max_samples is None:
            return n
        if isinstance(self.max_samples, float) and self.max_samples <= 1.0:
            return max(1, int(self.max_samples * n))
        return min(int(self.max_samples), n)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        self.classes_ = np.unique(y)
        n, C = len(X), len(self.classes_)
        rng = np.random.RandomState(self.seed)
        m = self._bootstrap_size(n)

        edges = make_bins(X)                      # chia khoảng một lần cho cả rừng
        Xb = bin_data(X, edges)

        y_idx = np.searchsorted(self.classes_, y)
        counts = np.bincount(y_idx, minlength=C)
        if self.class_weight == "balanced":       # trọng số tính trên toàn bộ tập train
            cw = np.where(counts > 0, n / (C * np.maximum(counts, 1)), 0.0)
        else:
            cw = np.ones(C)

        self.trees = []
        for i in range(self.n_estimators):
            idx = rng.randint(0, n, size=m)                       # bước 1: bootstrap
            tree = DecisionTreeScratch(max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf,
                                       max_features=self.max_features,   # bước 2
                                       seed=self.seed + i)
            tree.fit_binned(Xb[idx], y[idx], edges, classes=self.classes_, class_weights=cw)
            self.trees.append(tree)

        # độ quan trọng đặc trưng = trung bình của các cây
        self.feature_importances_ = np.mean([t.importances_ for t in self.trees], axis=0)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        rows = np.arange(len(X))
        for tree in self.trees:                                   # bước 3: mỗi cây bỏ 1 phiếu
            votes[rows, tree.predict_idx(X)] += 1
        return self.classes_[votes.argmax(axis=1)]                # label nhiều phiếu nhất


def choose_n_estimators(X_train, y_train, groups=None, stride=10, folds=3,
                        candidates=(25, 50, 100)):
    """
    CV theo nhóm trên 1/stride số frame (nhanh). Số cây càng nhiều càng ổn định nên chỉ
    chọn số lớn hơn khi điểm CV tăng rõ rệt.
    """
    if groups is None:
        print("  [CẢNH BÁO] Không có `groups` -> CV theo frame, điểm CV có thể cao giả do rò rỉ.")
    else:
        print(f"  CV theo nhóm: {len(np.unique(groups))} nhóm trong tập train, 1/{stride} số frame")
    best_n, best_cv = candidates[0], -1
    for n in candidates:
        cv = group_cv_score(lambda: RandomForestScratch(n_estimators=n),
                            X_train, y_train, groups, folds=folds, stride=stride)
        print(f"  n_estimators={n:3d} -> CV bal-acc = {cv:.4f}")
        if cv > best_cv + 0.002:
            best_n, best_cv = n, cv
    print(f"  => chọn n_estimators = {best_n}")
    return best_n


def run(X_train, X_test, y_train, y_test, groups_train=None, stride=10, feature_cols=None):
    print("\n=== RANDOM FOREST (tự code) ===")
    if groups_train is None:                       # lấy nhóm do split_data() của common.py lưu lại
        g = get_train_groups()
        if g is not None and len(g) == len(y_train):
            groups_train = g
    n = choose_n_estimators(X_train, y_train, groups_train, stride=stride)
    model = RandomForestScratch(n_estimators=n).fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"  Random Forest test accuracy          = {accuracy(y_test, pred):.4f}")
    print(f"  Random Forest test balanced accuracy = {balanced_accuracy(y_test, pred):.4f}")
    cols = feature_cols if feature_cols is not None else list(range(len(model.feature_importances_)))
    print("  Top 5 đặc trưng quan trọng:")
    for i in np.argsort(model.feature_importances_)[::-1][:5]:
        print(f"    {feature_name(cols[i]):32s} {model.feature_importances_[i]:.3f}")
    return model, pred


if __name__ == "__main__":
    LABEL = "full"        # "full" (gốc_loại, 56 lớp) | "root" (gốc) | "quality" (loại)
    FEATURES = "all"      # "all" | "left" (tay trái) | "right" (tay phải)

    X_train, X_test, y_train, y_test = split_data()[:4]
    y_train, y_test = convert_labels(y_train, LABEL), convert_labels(y_test, LABEL)
    X_train, cols = select_features(X_train, FEATURES)
    X_test, _ = select_features(X_test, FEATURES)
    print(f"Học theo: {LABEL} | đặc trưng: {FEATURES} ({len(cols)} cột) | "
          f"train={len(y_train)} test={len(y_test)} | {len(set(y_train))} lớp")
    run(X_train, X_test, y_train, y_test, feature_cols=cols)