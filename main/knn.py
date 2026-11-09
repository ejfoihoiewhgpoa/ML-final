"""
KNN (K-Nearest Neighbors) - tự code, chỉ dùng numpy (bản nhanh cho dữ liệu lớn)

Ý tưởng: KHÔNG "học" gì cả, chỉ nhớ toàn bộ tập train.
Dự đoán 1 mẫu mới:
  1) Tính khoảng cách Euclid từ mẫu đó tới MỌI mẫu train
  2) Lấy K mẫu gần nhất
  3) Bỏ phiếu: label xuất hiện nhiều nhất trong K mẫu đó thì thắng
     (nếu hòa phiếu, label của hàng xóm gần nhất thắng)
Siêu tham số chính: K. K nhỏ -> nhạy nhiễu, K lớn -> quá mịn.

Vì sao bản này nhanh hơn bản vòng lặp từng mẫu (cùng thuật toán, kết quả như nhau):
  - Khoảng cách tính cho CẢ LÔ mẫu cùng lúc bằng nhân ma trận, dùng công thức
        |a - b|^2 = |a|^2 + |b|^2 - 2 a.b
  - Khi chọn K, chỉ tính hàng xóm gần nhất MỘT lần với K lớn nhất (15), rồi lấy
    5, 7, 9... hàng xóm đầu tiên cho từng K, thay vì tính lại 7 lần.
  - Các frame liên tiếp trong video gần như giống nhau, nên khi chọn K chỉ dùng
    1 trong mỗi STRIDE_CV frame của tập train (đỡ tốn thời gian mà không đổi kết luận).

Chạy riêng: python knn.py
"""
import time

import numpy as np

from common import accuracy, get_train_groups, make_folds, split_data

KS = [1, 3, 5, 7, 9, 11, 15]   # các giá trị K để thử
FOLDS = 5
STRIDE_CV = 6      # chọn K: chỉ dùng 1/6 số frame của tập train
STRIDE_FIT = 2     # model cuối: dùng 1/2 số frame của tập train (1 = dùng hết, chậm hơn)
BATCH = 256        # số mẫu tính khoảng cách mỗi lượt (giảm nếu thiếu RAM)


class KNNScratch:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        # KNN chỉ lưu dữ liệu, không có bước học
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.classes_, self.y_idx = np.unique(np.asarray(y), return_inverse=True)
        self.sq = (self.X * self.X).sum(axis=1)          # |b|^2 của mọi mẫu train
        return self

    def _neighbors(self, Xq, kmax):
        """Chỉ số kmax hàng xóm gần nhất của từng mẫu truy vấn, sắp từ gần tới xa."""
        kmax = min(kmax, len(self.X))
        out = np.empty((len(Xq), kmax), dtype=np.int64)
        for s in range(0, len(Xq), BATCH):
            A = Xq[s:s + BATCH]
            # bước 1: bình phương khoảng cách Euclid tới mọi mẫu train, cả lô cùng lúc
            D = (A * A).sum(axis=1)[:, None] + self.sq[None, :] - 2.0 * (A @ self.X.T)
            # bước 2: lấy kmax mẫu nhỏ nhất (argpartition nhanh hơn sắp xếp toàn bộ)
            part = np.argpartition(D, kmax - 1, axis=1)[:, :kmax]
            order = np.argsort(np.take_along_axis(D, part, axis=1), axis=1)
            out[s:s + len(A)] = np.take_along_axis(part, order, axis=1)
        return out

    def predict_multi_k(self, X, ks):
        """Dự đoán cho nhiều giá trị K cùng lúc. Trả về dict {k: mảng label}."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        kmax = max(ks)
        nbr_labels = self.y_idx[self._neighbors(X, kmax)]       # (số mẫu, kmax) label hàng xóm
        counts = np.zeros((len(X), len(self.classes_)))
        rows = np.arange(len(X))
        preds = {}
        for j in range(nbr_labels.shape[1]):
            # bước 3: mỗi hàng xóm bỏ 1 phiếu, cộng thêm một lượng rất nhỏ giảm dần theo
            # độ xa (1e-3, 0.5e-3, 0.25e-3...). Hàng xóm gần nhất có lượng này lớn hơn
            # tổng của tất cả hàng xóm xa hơn -> nếu hòa phiếu thì label của hàng xóm
            # gần nhất thắng
            counts[rows, nbr_labels[:, j]] += 1.0 + 1e-3 * 0.5 ** j
            if (j + 1) in ks:
                preds[j + 1] = self.classes_[counts.argmax(axis=1)]
        return preds

    def predict(self, X):
        return self.predict_multi_k(X, [self.k])[self.k]


def choose_k(X_train, y_train):
    """Chọn K bằng cross-validation theo nhóm (sequence), trên một phần frame của tập train."""
    groups = get_train_groups()
    if groups is not None and len(groups) != len(y_train):
        groups = None
    sub = np.arange(0, len(y_train), STRIDE_CV)
    Xs, ys = X_train[sub], y_train[sub]
    gs = groups[sub] if groups is not None else None
    print(f"  Chọn K trên {len(sub)} frame (1/{STRIDE_CV} tập train), {FOLDS} fold theo nhóm")

    fold_id = make_folds(ys, FOLDS, groups=gs)
    scores = {k: [] for k in KS}
    for f in range(FOLDS):
        val = fold_id == f
        if not val.any() or val.all():
            continue
        t0 = time.time()
        model = KNNScratch().fit(Xs[~val], ys[~val])
        preds = model.predict_multi_k(Xs[val], KS)
        for k in KS:
            scores[k].append(accuracy(ys[val], preds[k]))
        print(f"    fold {f + 1}/{FOLDS} xong ({time.time() - t0:.0f}s)")

    best_k, best_cv = KS[0], -1
    for k in KS:
        cv = float(np.mean(scores[k]))
        print(f"  k={k:2d} -> CV acc = {cv:.4f}")
        if cv > best_cv:
            best_k, best_cv = k, cv
    print(f"  => chọn k = {best_k}")
    return best_k


def run(X_train, X_test, y_train, y_test):
    print("\nKNN")
    k = choose_k(X_train, y_train)

    t0 = time.time()
    model = KNNScratch(k).fit(X_train[::STRIDE_FIT], y_train[::STRIDE_FIT])
    print(f"  Dự đoán {len(X_test)} mẫu test với {len(model.X)} mẫu train (có thể mất vài phút)...")
    pred = model.predict(X_test)
    print(f"  KNN test acc = {accuracy(y_test, pred):.4f}  ({time.time() - t0:.0f}s)")
    return model, pred


if __name__ == "__main__":
    run(*split_data())