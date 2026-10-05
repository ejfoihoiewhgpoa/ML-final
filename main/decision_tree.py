import numpy as np

def gini(y):                                    # độ lẫn của một nhóm nhãn (0 = thuần)
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):                        # khác KNN: fit thật sự xây cây
        self.root = self._grow(np.array(X), np.array(y), 0)

    def _grow(self, X, y, depth):
        # dừng: đủ sâu hoặc nhóm đã thuần -> tạo lá = nhãn đa số
        if depth >= self.max_depth or gini(y) == 0:
            return np.bincount(y).argmax()

        best_gain, best_f, best_t = 0, None, None
        for f in range(X.shape[1]):             # thử từng đặc trưng
            for t in np.unique(X[:, f]):        # thử từng ngưỡng
                left = X[:, f] <= t
                if left.all():                  # nhánh phải rỗng -> bỏ
                    continue
                child = left.mean() * gini(y[left]) + (~left).mean() * gini(y[~left])
                gain = gini(y) - child          # giảm Gini bao nhiêu
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, t

        if best_f is None:                      # không chia được nữa
            return np.bincount(y).argmax()

        left = X[:, best_f] <= best_t
        return (best_f, best_t,                 # nút: (đặc trưng, ngưỡng, nhánh trái, nhánh phải)
                self._grow(X[left], y[left], depth + 1),
                self._grow(X[~left], y[~left], depth + 1))

    def _predict_one(self, x):
        node = self.root
        while isinstance(node, tuple):          # còn là nút thì đi tiếp, gặp lá thì dừng
            f, t, left, right = node
            node = left if x[f] <= t else right
        return node

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.array(X)])

# Dùng thử (giống ví dụ KNN)
X_train = [[1,2],[2,3],[5,5],[6,5]]
y_train = [0,0,1,1]
model = DecisionTree(max_depth=3)
model.fit(X_train, y_train)
print(model.predict([[1.5,1.5],[5.5,5]]))   # [0 1]