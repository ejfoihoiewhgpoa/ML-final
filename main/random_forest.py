import numpy as np

def gini(y):                                   # độ lẫn của một nhóm nhãn (0 = thuần)
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)

class Tree:
    def __init__(self, max_depth=5, n_features=None):
        self.max_depth = max_depth
        self.n_features = n_features           # số đặc trưng được xét mỗi lần chia

    def fit(self, X, y):
        self.root = self._grow(X, y, 0)

    def _grow(self, X, y, depth):
        # dừng: đủ sâu hoặc nhóm đã thuần -> lá = nhãn đa số
        if depth >= self.max_depth or gini(y) == 0:
            return np.bincount(y).argmax()

        d = X.shape[1]
        feats = np.random.choice(d, self.n_features or d, replace=False)
        best_gain, best_f, best_t = 0, None, None

        for f in feats:                        # thử từng đặc trưng
            for t in np.unique(X[:, f]):       # thử từng ngưỡng
                left = X[:, f] <= t
                if left.all():                 # nhánh phải rỗng -> bỏ
                    continue
                child = left.mean() * gini(y[left]) + (~left).mean() * gini(y[~left])
                gain = gini(y) - child         # giảm Gini bao nhiêu
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, f, t

        if best_f is None:                     # không chia được nữa
            return np.bincount(y).argmax()

        left = X[:, best_f] <= best_t
        return (best_f, best_t,
                self._grow(X[left], y[left], depth + 1),
                self._grow(X[~left], y[~left], depth + 1))

    def _predict_one(self, x):
        node = self.root
        while isinstance(node, tuple):         # còn là nút (chưa phải lá) thì đi tiếp
            f, t, left, right = node
            node = left if x[f] <= t else right
        return node

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])


class RandomForest:
    def __init__(self, n_trees=30, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        n, d = X.shape
        self.trees = []
        for _ in range(self.n_trees):
            idx = np.random.randint(0, n, n)           # bootstrap: lấy mẫu có hoàn lại
            tree = Tree(self.max_depth, n_features=max(1, int(np.sqrt(d))))
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        X = np.array(X)
        votes = np.array([t.predict(X) for t in self.trees])   # mỗi hàng: 1 cây
        return np.array([np.bincount(col).argmax() for col in votes.T])  # bỏ phiếu từng điểm

X_train = [[1,2],[2,3],[5,5],[6,5]]
y_train = [0,0,1,1]
model = RandomForest(n_trees=10)
model.fit(X_train, y_train)
print(model.predict([[1.5,1.5],[5.5,5]]))   # [0 1]