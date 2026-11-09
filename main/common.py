"""
COMMON: phần dùng chung, KHÔNG dùng sklearn.
  - đọc bảng danh mục train.csv (path, file_id, sequence_id, participant_id, root, quality, label)
    rồi mở từng file parquet theo cột `path` để lấy landmark
  - chuẩn hóa landmark (dùng khi chạy webcam, hoặc khi dữ liệu chưa chuẩn hóa)
  - chia train/test và cross-validation THEO NHÓM (sequence), accuracy, confusion matrix, report

Vì sao chia theo nhóm: các frame trong cùng một chuỗi (video) gần như giống nhau.
Nếu frame này ở train, frame kế bên ở test thì model chỉ cần "nhớ" là đoán đúng
-> accuracy cao giả. Chia theo sequence_id thì cả chuỗi nằm trọn ở train HOẶC test.

Cần cài: pip install pyarrow   (để pandas đọc được parquet)
"""
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# CẤU HÌNH - sửa cho khớp với dữ liệu của bạn
# ---------------------------------------------------------------------
META_CSV = r"D:\machine_learning\final_project\progress\my_dataset\train_landmarks\train.csv"
DATA_ROOT = None        # thư mục gốc để nối với cột `path` nếu path là đường dẫn tương đối.
                        # None = thư mục chứa META_CSV
SEARCH_ROOT = None      # nếu không thấy file theo `path`, tự quét thư mục này để tìm.
                        # None = quét thư mục cha của thư mục `my_dataset` (tức `progress`)
SKIP_MISSING = True     # True: bỏ qua file parquet không tìm thấy (in báo cáo thiếu); False: báo lỗi ngay
USE_CACHE = True        # lưu kết quả đọc vào landmarks_cache.npz, lần sau chạy gần như tức thì.
                        # Xóa file đó nếu muốn đọc lại từ parquet (đổi cấu hình thì tự đọc lại)
PATH_COL = "path"       # cột chứa đường dẫn tới file parquet
LABEL_COL = "label"     # cột nhãn trong META_CSV
GROUP_COL = "sequence_id"   # chia theo nhóm này. Đổi thành "participant_id" nếu muốn
                            # test trên NGƯỜI CHƯA TỪNG THẤY (khó hơn, sát thực tế hơn)
DROP_COLS = []          # chỉ dùng khi parquet KHÔNG có cột tên x_left_hand_0... (đọc kiểu tổng quát)
DROP_EMPTY_FRAMES = True    # bỏ frame mà CẢ HAI tay đều không được nhận ra
EXTRA_POSITION = True       # thêm 9 đặc trưng VỊ TRÍ tuyệt đối của 2 cổ tay (chuẩn hóa từng tay đã xóa vị trí)
POS_WEIGHT = 4.0            # nhân đặc trưng vị trí lên để không bị 126 đặc trưng hình dạng lấn át
MIN_FRAMES_PER_CLASS = 100  # bỏ class có ít frame hơn mức này (không đủ dữ liệu để học / đánh giá)
FEATURE_VERSION = "hands-v4"   # đổi chuỗi này để buộc đọc lại dữ liệu (không dùng cache cũ)

SEED = 42
TEST_SIZE = 0.2

_TRAIN_GROUPS = None    # split_data() lưu nhóm của tập train ở đây để cross_val_score tự dùng


# ======================================================================
# ĐẶC TRƯNG HAI TAY
# ----------------------------------------------------------------------
# Dữ liệu: mỗi frame có 21 điểm x 3 tọa độ (x, y, z) cho tay trái và tay phải.
# Tay không được nhận ra -> toàn bộ cột của tay đó là NaN.
# Với MỖI tay:
#   1) Dời gốc về cổ tay (điểm 0)            -> bỏ ảnh hưởng vị trí tay trong khung hình
#   2) Chia cho kích thước bàn tay = trung bình khoảng cách 3D từ cổ tay tới 4 khớp gốc
#      ngón (điểm 5, 9, 13, 17) -> bỏ ảnh hưởng xa/gần. Dùng 4 điểm và cả trục z để
#      kích thước không về ~0 khi bàn tay hướng thẳng vào camera (gây giá trị nổ)
#      Sau đó chặn mọi giá trị vào [-5, 5] để vài frame nhận diện lỗi không phá khoảng cách
#   3) Tay vắng mặt -> điền 0 (NaN làm hỏng phép tính khoảng cách)
# Đặc trưng cuối: [tay trái 63] + [tay phải 63] + [trái có mặt?, phải có mặt?] = 128 số
#   (+ 9 số vị trí nếu EXTRA_POSITION: cổ tay trái xyz, cổ tay phải xyz, phải - trái)
# Cột 126, 127 luôn là 2 cờ "có mặt" (xem FLAG_COLS)
# ======================================================================
HANDS = ("left", "right")
CLIP = 5.0
FLAG_COLS = slice(126, 128)     # 2 cột cờ "tay trái / tay phải có mặt"


def _normalize_hands(P):
    """P: (n, 21, 3) với NaN nếu tay vắng. Trả về (n, 63) đã chuẩn hóa và cờ có mặt (n,)."""
    P = np.asarray(P, dtype=np.float32)
    missing = np.isnan(P).any(axis=(1, 2))
    P = np.nan_to_num(P)
    P = P - P[:, :1, :]                               # bước 1: gốc tại cổ tay
    scale = np.linalg.norm(P[:, [5, 9, 13, 17], :], axis=2).mean(axis=1)   # bước 2: kích thước tay
    scale[scale < 1e-6] = 1.0
    P = np.clip(P / scale[:, None, None], -CLIP, CLIP)
    P[missing] = 0.0                                  # bước 3: tay vắng -> 0
    return P.reshape(len(P), 63), (~missing).astype(np.float32)


def _position_features(left, right):
    """Vị trí tuyệt đối của cổ tay mỗi tay (x, y, z) và độ lệch phải - trái. Tay vắng -> 0."""
    lw = np.asarray(left, dtype=np.float32)[:, 0, :]
    rw = np.asarray(right, dtype=np.float32)[:, 0, :]
    lm, rm = np.isnan(lw).any(axis=1), np.isnan(rw).any(axis=1)
    diff = np.where((lm | rm)[:, None], 0.0, rw - lw)
    return POS_WEIGHT * np.hstack([np.nan_to_num(lw), np.nan_to_num(rw), diff]).astype(np.float32)


def build_features(left, right):
    """left, right: (n, 21, 3). Trả về ma trận (n, 128) hoặc (n, 137) nếu EXTRA_POSITION."""
    L, lp = _normalize_hands(left)
    R, rp = _normalize_hands(right)
    feats = [L, R, lp[:, None], rp[:, None]]
    if EXTRA_POSITION:
        feats.append(_position_features(left, right))
    return np.hstack(feats)


def normalize(left_landmarks, right_landmarks):
    """
    Dùng khi chạy webcam với MediaPipe Holistic: truyền results.left_hand_landmarks.landmark
    và results.right_hand_landmarks.landmark (hoặc None nếu tay đó không thấy).
    Trả về vector 128 số, GIỐNG HỆT cách dữ liệu train được tạo.
    """
    def to_array(lms):
        if lms is None:
            return np.full((1, 21, 3), np.nan, dtype=np.float32)
        return np.array([[[p.x, p.y, p.z] for p in lms]], dtype=np.float32)
    return build_features(to_array(left_landmarks), to_array(right_landmarks))[0]


# ======================================================================
# ĐỌC DỮ LIỆU: train.csv (danh mục) + các file parquet
# ======================================================================
_INDEX = None        # (tên thư mục cha, tên file) -> đường dẫn
_INDEX_NAME = None   # tên file -> danh sách đường dẫn


def _build_index(search_root):
    by_key, by_name = {}, {}
    for dirpath, _, files in os.walk(search_root):
        for fn in files:
            if fn.endswith(".parquet"):
                full = os.path.join(dirpath, fn)
                by_key.setdefault((os.path.basename(dirpath), fn), full)
                by_name.setdefault(fn, []).append(full)
    return by_key, by_name


def _resolve(path, root):
    """
    Tìm file thật. Thứ tự thử:
      1) đường dẫn trong csv (tuyệt đối, hoặc nối với DATA_ROOT)
      2) khớp (tên thư mục cha, tên file) trong SEARCH_ROOT
      3) khớp riêng tên file, nếu tên đó là DUY NHẤT trong SEARCH_ROOT
    Không thấy -> trả về None.
    """
    global _INDEX, _INDEX_NAME
    path = str(path)
    for p in (path, path.replace("\\", os.sep).replace("/", os.sep)):
        if os.path.exists(p):
            return p
        q = os.path.join(root, p)
        if os.path.exists(q):
            return q

    if _INDEX is None:
        search_root = SEARCH_ROOT or os.path.dirname(os.path.dirname(os.path.dirname(META_CSV)))
        print(f"Không thấy file theo đường dẫn trong csv, đang quét {search_root} ...")
        _INDEX, _INDEX_NAME = _build_index(search_root)
        print(f"  tìm thấy {len(_INDEX_NAME)} tên file .parquet")
    parts = path.replace("\\", "/").split("/")
    fn = parts[-1]
    key = (parts[-2] if len(parts) > 1 else "", fn)
    if key in _INDEX:
        return _INDEX[key]
    if len(_INDEX_NAME.get(fn, [])) == 1:      # tên file duy nhất -> chắc chắn đúng file đó
        return _INDEX_NAME[fn][0]
    return None


def _features_from_df(d):
    """Đặc trưng của mọi frame trong 1 file parquet. Trả về ma trận (số frame, số đặc trưng)."""
    if "x_left_hand_0" in d.columns:
        blocks = []
        for side in HANDS:
            cols = [f"{ax}_{side}_hand_{i}" for i in range(21) for ax in "xyz"]
            blocks.append(d[cols].to_numpy(dtype=np.float32).reshape(len(d), 21, 3))
        X = build_features(*blocks)
        if DROP_EMPTY_FRAMES:
            X = X[X[:, FLAG_COLS].sum(axis=1) > 0]       # còn ít nhất 1 tay
        return X

    # định dạng khác: lấy mọi cột số (trừ DROP_COLS), hoặc 1 cột chứa cả mảng
    d = d.drop(columns=[c for c in DROP_COLS if c in d.columns])
    feats = []
    for c in d.columns:
        first = d[c].iloc[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            feats.append(np.stack(d[c].values).astype(float))
        elif np.issubdtype(d[c].dtype, np.number):
            feats.append(d[[c]].values.astype(float))
    return np.hstack(feats)


CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "landmarks_cache.npz")


def _signature():
    return "|".join(map(str, [os.path.getmtime(META_CSV), DROP_COLS, FEATURE_VERSION, DROP_EMPTY_FRAMES, EXTRA_POSITION, POS_WEIGHT, MIN_FRAMES_PER_CLASS,
                              LABEL_COL, GROUP_COL, SKIP_MISSING, SEARCH_ROOT, DATA_ROOT]))


def load_data(return_groups=False):
    sig = _signature()
    if USE_CACHE and os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH, allow_pickle=False)
        if str(z["sig"]) == sig:
            print(f"Đọc dữ liệu từ cache {os.path.basename(CACHE_PATH)} (xóa file này nếu muốn đọc lại parquet)")
            X, y, g = z["X"], z["y"], z["g"]
            return (X, y, g) if return_groups else (X, y)

    meta = pd.read_csv(META_CSV)
    root = DATA_ROOT or os.path.dirname(META_CSV)

    Xs, ys, gs = [], [], []
    missing = []
    for i, (_, row) in enumerate(meta.iterrows()):
        if i and i % 500 == 0:
            print(f"  đã xử lý {i}/{len(meta)} dòng...")
        p = _resolve(row[PATH_COL], root)
        if p is None:
            if not SKIP_MISSING:
                raise FileNotFoundError(f"Không thấy file parquet: {row[PATH_COL]}")
            missing.append(row)
            continue
        d = pd.read_parquet(p)
        if len(d) == 0:
            continue
        feat = _features_from_df(d)
        Xs.append(feat)
        ys.extend([str(row[LABEL_COL])] * len(feat))
        gs.extend([str(row[GROUP_COL])] * len(feat))     # mọi frame của 1 file chung 1 nhóm

    if missing:
        miss = pd.DataFrame(missing)
        print(f"[Chú ý] Thiếu {len(miss)}/{len(meta)} file parquet ({len(miss) / len(meta):.1%}), đã bỏ qua.")
        total = meta[LABEL_COL].value_counts()
        lost = miss[LABEL_COL].value_counts()
        print("  Số file thiếu / tổng, theo label (10 label thiếu nhiều nhất):")
        for lab, n in lost.head(10).items():
            print(f"    {lab:8s} {n}/{total[lab]}")
        print("  Ví dụ file thiếu:", list(miss[PATH_COL].head(3)))
    if not Xs:
        raise FileNotFoundError("Không đọc được file parquet nào. Kiểm tra SEARCH_ROOT / cột path.")

    X = np.vstack(Xs)
    if np.isnan(X).any():
        print(f"[Chú ý] Còn {int(np.isnan(X).sum())} giá trị NaN trong đặc trưng -> thay bằng 0")
        X = np.nan_to_num(X)
    y, g = np.array(ys), np.array(gs)

    labels, counts = np.unique(y, return_counts=True)
    small = labels[counts < MIN_FRAMES_PER_CLASS]
    if len(small):
        keep = ~np.isin(y, small)
        print(f"[Chú ý] Bỏ {len(small)} class có dưới {MIN_FRAMES_PER_CLASS} frame: {list(map(str, small))}")
        X, y, g = X[keep], y[keep], g[keep]
    if USE_CACHE:
        np.savez(CACHE_PATH, X=X, y=y, g=g, sig=np.array(sig))
    return (X, y, g) if return_groups else (X, y)


# ======================================================================
# CHIA TRAIN / TEST THEO NHÓM (tự code)
# ======================================================================
def train_test_split(X, y, groups=None, test_size=TEST_SIZE, seed=SEED):
    """
    Nếu có groups: với MỖI class, chọn ngẫu nhiên khoảng test_size số NHÓM đưa vào test
    (cả nhóm vào test, không xé lẻ frame). Nếu không có groups: chia từng mẫu như bình thường.
    Trả về X_train, X_test, y_train, y_test, g_train, g_test (g là None nếu không có groups).
    """
    rng = np.random.RandomState(seed)
    if groups is None:
        groups = np.arange(len(y)).astype(str)     # mỗi mẫu là 1 nhóm riêng
        no_groups = True
    else:
        no_groups = False

    test_groups = set()
    for c in np.unique(y):
        gs = np.unique(groups[y == c])
        if len(gs) < 2:
            continue                                # class chỉ có 1 nhóm: giữ hết ở train
        n_test = max(1, int(round(len(gs) * test_size)))
        already = [g for g in gs if g in test_groups]
        need = n_test - len(already)
        cand = np.array([g for g in gs if g not in test_groups])
        need = min(need, len(cand) - 1)             # luôn chừa ít nhất 1 nhóm cho train
        if need > 0:
            test_groups.update(rng.choice(cand, need, replace=False))

    test_mask = np.array([g in test_groups for g in groups])
    tr, te = np.where(~test_mask)[0], np.where(test_mask)[0]
    rng.shuffle(tr)
    rng.shuffle(te)
    g_tr = None if no_groups else groups[tr]
    g_te = None if no_groups else groups[te]
    return X[tr], X[te], y[tr], y[te], g_tr, g_te


def split_data(verbose=True):
    global _TRAIN_GROUPS
    X, y, g = load_data(return_groups=True)
    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(X, y, g)
    _TRAIN_GROUPS = g_train
    if verbose:
        print(f"Tổng {len(X)} mẫu (frame), {len(np.unique(y))} class, {X.shape[1]} đặc trưng, "
              f"{len(np.unique(g))} nhóm theo '{GROUP_COL}'")
        print(f"Train: {len(X_train)} mẫu / {len(np.unique(g_train))} nhóm | "
              f"Test: {len(X_test)} mẫu / {len(np.unique(g_test))} nhóm")
        missing = set(np.unique(y)) - set(np.unique(y_test))
        if missing:
            print(f"[Chú ý] {len(missing)} class không có mẫu nào ở tập test: {sorted(missing)[:8]}...")
    return X_train, X_test, y_train, y_test


# ======================================================================
# CROSS-VALIDATION THEO NHÓM (tự code)
# ======================================================================
def get_train_groups():
    """Nhóm (sequence) của tập train, do split_data() lưu lại. Dùng để chia fold theo nhóm."""
    return _TRAIN_GROUPS


def make_folds(y, folds=5, seed=SEED, groups=None):
    """
    Trả về mảng fold_id (0..folds-1) cho từng mẫu. Nếu có nhóm thì mỗi nhóm nằm trọn
    trong 1 fold; mỗi class được rải đều qua các fold.
    """
    if groups is None and _TRAIN_GROUPS is not None and len(_TRAIN_GROUPS) == len(y):
        groups = _TRAIN_GROUPS
    if groups is None:
        groups = np.arange(len(y)).astype(str)

    rng = np.random.RandomState(seed)
    group_fold = {}
    for c in np.unique(y):
        gs = np.unique(groups[y == c])
        rng.shuffle(gs)
        offset = rng.randint(folds)
        for i, g in enumerate(gs):
            if g not in group_fold:
                group_fold[g] = (i + offset) % folds
    return np.array([group_fold[g] for g in groups])


def cross_val_score(make_model, X, y, folds=5, seed=SEED, groups=None):
    """
    Chia tập train thành `folds` phần. Lần lượt lấy 1 phần làm validation, train trên
    các phần còn lại, rồi lấy trung bình accuracy. Dùng để chọn siêu tham số
    (K, max_depth...) mà không đụng tới tập test.

    Nếu có nhóm (mặc định lấy từ split_data) thì mỗi nhóm nằm trọn trong 1 fold.
    make_model: hàm không tham số, trả về 1 model mới (có .fit và .predict).
    """
    fold_id = make_folds(y, folds, seed, groups)
    scores = []
    for f in range(folds):
        val = fold_id == f
        if not val.any() or val.all():
            continue
        model = make_model().fit(X[~val], y[~val])
        scores.append(accuracy(y[val], model.predict(X[val])))
    return float(np.mean(scores))


# ======================================================================
# METRICS (tự code)
# ======================================================================
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, classes):
    """cm[i][j] = số mẫu thật sự là class i nhưng bị đoán là class j."""
    pos = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[pos[t], pos[p]] += 1
    return cm


def classification_report(y_true, y_pred, classes):
    """
    precision = đoán đúng class đó / tổng số lần đoán class đó
    recall    = đoán đúng class đó / tổng số mẫu thật của class đó
    f1        = trung bình điều hòa của precision và recall
    """
    cm = confusion_matrix(y_true, y_pred, classes)
    tp = np.diag(cm).astype(float)
    predicted, support = cm.sum(0), cm.sum(1)
    precision, recall = np.zeros(len(classes)), np.zeros(len(classes))
    np.divide(tp, predicted, out=precision, where=predicted > 0)
    np.divide(tp, support, out=recall, where=support > 0)
    denom = precision + recall
    f1 = np.zeros(len(classes))
    np.divide(2 * precision * recall, denom, out=f1, where=denom > 0)

    print(f"{'class':8s} {'precision':>9s} {'recall':>8s} {'f1':>8s} {'support':>8s}")
    for i, c in enumerate(classes):
        print(f"{str(c):8s} {precision[i]:9.3f} {recall[i]:8.3f} {f1[i]:8.3f} {support[i]:8d}")
    print(f"{'macro':8s} {precision.mean():9.3f} {recall.mean():8.3f} {f1.mean():8.3f} {support.sum():8d}")