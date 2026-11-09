"""
DECISION TREE (Cây quyết định) - tự code, chỉ dùng numpy (bản nhanh cho dữ liệu lớn)

Ý tưởng: xây 1 cây gồm các câu hỏi dạng "đặc trưng f_i <= ngưỡng t ?".

Gini impurity của 1 nhóm mẫu:  G = 1 - sum_c (p_c)^2
  (p_c = tỉ lệ class c trong nhóm; G = 0 nghĩa là nhóm thuần 1 class)

Cách xây (đệ quy từ gốc xuống):
  1) Với mỗi đặc trưng, thử các ngưỡng chia. Mỗi ngưỡng chia mẫu thành nhóm trái (<= t)
     và nhóm phải (> t). Điểm của ngưỡng = Gini trung bình có trọng số của 2 nhóm.
  2) Chọn (đặc trưng, ngưỡng) có điểm thấp nhất, chia dữ liệu thành 2 nhánh
  3) Lặp lại cho từng nhánh. Dừng (tạo lá) khi: nhánh thuần 1 class, hoặc đạt max_depth,
     hoặc quá ít mẫu (min_samples_split / min_samples_leaf), hoặc không có cách chia hợp lệ.
  Lá trả về class chiếm đa số trong lá đó.
Dự đoán: mẫu mới đi từ gốc xuống lá theo các câu hỏi.

Để chạy nhanh với ~275k frame (cùng thuật toán, chỉ khác cách tìm ngưỡng):
  - HISTOGRAM: trước khi xây cây, mỗi đặc trưng được chia thành tối đa 32 khoảng theo phân vị
    (mỗi khoảng chứa số mẫu gần bằng nhau). Ngưỡng chỉ thử tại ranh giới các khoảng, và số mẫu
    mỗi class trong từng khoảng đếm bằng np.bincount (1 lượt duyệt dữ liệu), rồi cộng dồn
    (cumsum) để biết ngay nhóm trái / phải của MỌI ngưỡng. Bản cũ phải sắp xếp lại dữ liệu ở mỗi nút.
  - class_weight="balanced": mẫu của class ít frame được tính trọng số lớn hơn, để class
    nhiều frame không lấn át (các class của bạn lệch từ ~100 đến ~9000 frame).

Chạy riêng: python decision_tree.py
"""
import numpy as np

from common import (SEED, accuracy, get_train_groups, make_folds, split_data)

N_BINS = 32
LEFT_COLS = list(range(0, 63)) + [126]         # tay trái (hình dạng) + cờ tay trái có mặt
RIGHT_COLS = list(range(63, 126)) + [127]      # tay phải (hình dạng) + cờ tay phải có mặt


# ======================================================================
# TIỆN ÍCH DÙNG CHUNG (random_forest.py cũng import từ đây)
# ======================================================================
def convert_labels(y, mode="full"):
    """full: giữ nguyên (vd C_R) | root: phần trước dấu _ (C) | quality: phần sau (R)."""
    y = np.asarray(y).astype(str)
    if mode == "full":
        return y
    if mode == "root":
        return np.array([s.split("_")[0] for s in y])
    if mode == "quality":
        return np.array([s.split("_", 1)[-1] for s in y])
    raise ValueError("mode phải là 'full', 'root' hoặc 'quality'")


def select_features(X, which="all"):
    """Trả về (X đã chọn cột, danh sách chỉ số cột gốc). which: all | left | right."""
    cols = {"all": list(range(X.shape[1])), "left": LEFT_COLS, "right": RIGHT_COLS}[which]
    return X[:, cols], cols


def feature_name(i):
    """Tên dễ đọc của cột đặc trưng thứ i (theo bố cục của common.py)."""
    if i < 126:
        hand = "trái" if i < 63 else "phải"
        j = i % 63
        return f"tay {hand}, điểm {j // 3}, trục {'xyz'[j % 3]}"
    if i < 128:
        return "cờ: tay trái có mặt" if i == 126 else "cờ: tay phải có mặt"
    pos = ["cổ tay trái x", "cổ tay trái y", "cổ tay trái z",
           "cổ tay phải x", "cổ tay phải y", "cổ tay phải z",
           "lệch phải-trái x", "lệch phải-trái y", "lệch phải-trái z"]
    return "vị trí: " + pos[i - 128] if i - 128 < len(pos) else f"cột {i}"


def balanced_accuracy(y_true, y_pred):
    """Trung bình recall của từng class: mỗi class đóng góp như nhau dù nhiều hay ít frame."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = [np.mean(y_pred[y_true == c] == c) for c in np.unique(y_true)]
    return float(np.mean(recalls))


def group_cv_score(make_model, X, y, groups=None, folds=3, stride=10, seed=SEED):
    """
    Cross-validation THEO NHÓM (mỗi sequence nằm trọn trong 1 fold) trên 1/stride số frame,
    trả về balanced accuracy trung bình.
    """
    idx = np.arange(0, len(y), stride)
    Xs, ys = X[idx], y[idx]
    gs = groups[idx] if groups is not None else None
    fold_id = make_folds(ys, folds, seed, groups=gs)
    scores = []
    for f in range(folds):
        val = fold_id == f
        if not val.any() or val.all():
            continue
        model = make_model().fit(Xs[~val], ys[~val])
        scores.append(balanced_accuracy(ys[val], model.predict(Xs[val])))
    return float(np.mean(scores))


# ======================================================================
# CHIA ĐẶC TRƯNG THÀNH CÁC KHOẢNG (HISTOGRAM)
# ======================================================================
def make_bins(X, n_bins=N_BINS):
    """Ranh giới các khoảng cho từng đặc trưng, theo phân vị. Trả về list mảng (float32)."""
    X = np.asarray(X, dtype=np.float32)
    q = np.percentile(X, np.linspace(0, 100, n_bins + 1)[1:-1], axis=0)   # (n_bins-1, số cột)
    return [np.unique(q[:, j].astype(np.float32)) for j in range(X.shape[1])]


def bin_data(X, edges):
    """Đổi mỗi giá trị thành chỉ số khoảng (0..31). x <= edges[b]  <=>  chỉ số <= b."""
    X = np.asarray(X, dtype=np.float32)
    Xb = np.empty(X.shape, dtype=np.uint8)
    for j, e in enumerate(edges):
        Xb[:, j] = np.searchsorted(e, X[:, j], side="left")
    return Xb


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, value):
        self.value = value          # chỉ số class chiếm đa số tại node này
        self.feature = None         # None => đây là lá
        self.threshold = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return self.left is None


class DecisionTreeScratch:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, class_weight=None, n_bins=N_BINS, seed=SEED):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features      # None | "sqrt" | int (Random Forest dùng "sqrt")
        self.class_weight = class_weight      # None | "balanced"
        self.n_bins = n_bins
        self.seed = seed

    # ------------------------------------------------------------------
    # HUẤN LUYỆN
    # ------------------------------------------------------------------
    def fit(self, X, y, classes=None):
        X = np.asarray(X, dtype=np.float32)
        edges = make_bins(X, self.n_bins)
        return self.fit_binned(bin_data(X, edges), y, edges, classes)

    def fit_binned(self, Xb, y, edges, classes=None, class_weights=None):
        """
        Train trên dữ liệu đã chia khoảng sẵn. Random Forest chia khoảng MỘT lần cho cả rừng
        rồi gọi hàm này cho từng cây, đỡ phải chia lại 100 lần.
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y) if classes is None else np.asarray(classes)
        y_idx = np.searchsorted(self.classes_, y)         # đổi label chuỗi -> số 0..C-1
        self.n_classes = len(self.classes_)
        self.n_features = Xb.shape[1]
        self.edges = edges
        self.rng = np.random.RandomState(self.seed)
        self.importances_ = np.zeros(self.n_features)

        if class_weights is None:
            counts = np.bincount(y_idx, minlength=self.n_classes)
            if self.class_weight == "balanced":
                present = (counts > 0).sum()
                class_weights = np.where(counts > 0, len(y_idx) / (present * np.maximum(counts, 1)), 0.0)
            else:
                class_weights = np.ones(self.n_classes)
        w = np.asarray(class_weights, dtype=float)[y_idx]  # trọng số của từng mẫu

        self.root = self._build(Xb, y_idx, w, depth=0)

        total = self.importances_.sum()
        if total > 0:
            self.importances_ /= total
        return self

    def _n_candidate_features(self):
        if self.max_features is None:
            return self.n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features)))
        return min(int(self.max_features), self.n_features)

    def _build(self, Xb, y, w, depth):
        n = len(y)
        wc = np.bincount(y, weights=w, minlength=self.n_classes)   # số mẫu (có trọng số) mỗi class
        W = wc.sum()
        node = Node(value=int(wc.argmax()))
        gini = 1.0 - ((wc / W) ** 2).sum() if W > 0 else 0.0

        # điều kiện dừng -> trả về lá
        if (gini <= 1e-12 or n < self.min_samples_split or n < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node

        # chọn các đặc trưng được phép xét ở lần chia này
        k = self._n_candidate_features()
        feats = (self.rng.choice(self.n_features, k, replace=False)
                 if k < self.n_features else np.arange(self.n_features))

        f, b, score = self._best_split(Xb, y, w, feats, wc, W)
        if f is None:                                     # không có cách chia hợp lệ
            return node

        mask = Xb[:, f] <= b
        self.importances_[f] += W * (gini - score)        # mức giảm Gini nhân tổng trọng số
        node.feature = int(f)
        node.threshold = float(self.edges[f][b])          # ngưỡng thật, dùng lúc dự đoán
        node.left = self._build(Xb[mask], y[mask], w[mask], depth + 1)
        node.right = self._build(Xb[~mask], y[~mask], w[~mask], depth + 1)
        return node

    def _best_split(self, Xb, y, w, feats, wc, W):
        """
        Với mỗi đặc trưng: đếm số mẫu (có trọng số) của từng class trong từng khoảng bằng bincount,
        cộng dồn theo khoảng -> số mẫu nhóm trái cho mọi ngưỡng, nhóm phải = tổng - trái.
        Trả về (đặc trưng, chỉ số khoảng ngưỡng, Gini trung bình có trọng số) tốt nhất.
        """
        n, C, B = len(y), self.n_classes, self.n_bins
        msl = self.min_samples_leaf
        best_f, best_b, best_score = None, None, np.inf

        for f in feats:
            n_edges = len(self.edges[f])
            if n_edges == 0:                              # đặc trưng hằng số
                continue
            col = Xb[:, f].astype(np.int64)
            hist = np.bincount(col * C + y, weights=w, minlength=B * C).reshape(B, C)
            cnt = np.bincount(col, minlength=B)

            left = np.cumsum(hist, axis=0)[:n_edges]      # ngưỡng b: nhóm trái = các khoảng <= b
            nl = np.cumsum(cnt)[:n_edges]
            nr = n - nl
            valid = (nl >= msl) & (nr >= msl)
            if not valid.any():
                continue

            right = wc - left
            wl, wr = left.sum(axis=1), right.sum(axis=1)
            gini_l = 1.0 - ((left / np.maximum(wl, 1e-12)[:, None]) ** 2).sum(axis=1)
            gini_r = 1.0 - ((right / np.maximum(wr, 1e-12)[:, None]) ** 2).sum(axis=1)
            score = np.where(valid, (wl * gini_l + wr * gini_r) / W, np.inf)

            b = int(score.argmin())
            if score[b] < best_score:
                best_f, best_b, best_score = f, b, score[b]
        return best_f, best_b, best_score

    # ------------------------------------------------------------------
    # DỰ ĐOÁN
    # ------------------------------------------------------------------
    def predict_idx(self, X):
        """Trả về chỉ số class (0..C-1) cho từng mẫu."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        out = np.empty(len(X), dtype=int)
        self._route(self.root, X, np.arange(len(X)), out)
        return out

    def _route(self, node, X, idx, out):
        """Đẩy cả nhóm mẫu xuống cây cùng lúc: mỗi node tách nhóm thành trái / phải."""
        if len(idx) == 0:
            return
        if node.is_leaf():
            out[idx] = node.value
            return
        go_left = X[idx, node.feature] <= node.threshold
        self._route(node.left, X, idx[go_left], out)
        self._route(node.right, X, idx[~go_left], out)

    def predict(self, X):
        return self.classes_[self.predict_idx(X)]

    def get_depth(self):
        def d(node):
            return 0 if node.is_leaf() else 1 + max(d(node.left), d(node.right))
        return d(self.root)

    def get_n_leaves(self):
        def c(node):
            return 1 if node.is_leaf() else c(node.left) + c(node.right)
        return c(self.root)


def choose_depth(X_train, y_train, groups=None, stride=10, folds=3,
                 candidates=(6, 8, 10, 12, 14, 18)):
    """CV theo nhóm, ưu tiên cây nông hơn nếu điểm không hơn rõ rệt (đỡ overfit)."""
    if groups is None:
        print("  [CẢNH BÁO] Không có `groups` -> CV theo frame, điểm CV có thể cao giả do rò rỉ.")
    else:
        print(f"  CV theo nhóm: {len(np.unique(groups))} nhóm trong tập train, 1/{stride} số frame")
    best_d, best_cv = candidates[0], -1
    for d in candidates:
        cv = group_cv_score(lambda: DecisionTreeScratch(max_depth=d, min_samples_leaf=20,
                                                        class_weight="balanced"),
                            X_train, y_train, groups, folds=folds, stride=stride)
        print(f"  max_depth={d:2d} -> CV bal-acc = {cv:.4f}")
        if cv > best_cv + 0.002:
            best_d, best_cv = d, cv
    print(f"  => chọn max_depth = {best_d}")
    return best_d


def run(X_train, X_test, y_train, y_test, groups_train=None, stride=10, feature_cols=None):
    print("\n=== DECISION TREE (tự code) ===")
    if groups_train is None:                       # lấy nhóm do split_data() của common.py lưu lại
        g = get_train_groups()
        if g is not None and len(g) == len(y_train):
            groups_train = g
    depth = choose_depth(X_train, y_train, groups_train, stride=stride)
    model = DecisionTreeScratch(max_depth=depth, min_samples_leaf=20,
                                class_weight="balanced").fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"  Độ sâu thực tế: {model.get_depth()} | Số lá: {model.get_n_leaves()}")
    print(f"  Decision Tree test accuracy          = {accuracy(y_test, pred):.4f}")
    print(f"  Decision Tree test balanced accuracy = {balanced_accuracy(y_test, pred):.4f}")
    cols = feature_cols if feature_cols is not None else list(range(len(model.importances_)))
    print("  Top 5 đặc trưng quan trọng:")
    for i in np.argsort(model.importances_)[::-1][:5]:
        print(f"    {feature_name(cols[i]):32s} {model.importances_[i]:.3f}")
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