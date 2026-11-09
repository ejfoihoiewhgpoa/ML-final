"""
APP DEMO: phát nhạc hợp âm bằng cử chỉ tay qua webcam.

Cách điều khiển:
  - TAY TRÁI (hình dạng) chọn LOẠI hợp âm: major, 7, maj7, minor, m7 (model từ train_quality.py)
  - PHÍM A-G chọn GỐC hợp âm (gốc không đoán được từ cử chỉ nên chọn bằng phím)
  - Hợp âm = gốc + loại, ví dụ gốc C + loại maj7 -> Cmaj7 -> phát file nhạc Cmaj7
  - X: đổi tay (nếu app đang dùng nhầm tay phải thay vì tay trái)
  - M: tắt / bật tiếng
  - ESC hoặc Q: thoát

Chạy:
  python train_quality.py     (một lần, để có quality_model.pkl)
  python app.py
Tùy chọn: --source 1 (webcam khác) | --source video.mp4 | --sounds "D:\\duong\\dan\\sounds"
"""
import argparse
import json
import pickle
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common                                         # noqa: E402  (normalize: cùng cách chuẩn hóa lúc train)
import decision_tree, knn, random_forest              # noqa: E402,F401  (cần để pickle nạp được model)

# ----------------------------------------------------------------------
# CẤU HÌNH - sửa cho khớp với bộ file nhạc của bạn
# ----------------------------------------------------------------------
MODEL_PATH = HERE / "quality_model.pkl"
ROOTS = list("ABCDEFG")
# Loại hợp âm trong dữ liệu -> hậu tố trong tên file nhạc. Đây là suy đoán từ danh sách file
# (A, A7, Am, Am7, Amaj7): hãy kiểm tra và sửa nếu sai (nhất là "R").
QUALITY_SUFFIX = {"1": "", "D": "7", "M": "maj7", "m1": "m", "R": "m7"}
QUALITY_DESC = {"1": "major", "D": "dominant 7", "M": "major 7", "m1": "minor", "R": "minor 7 (?)"}
SOUND_EXTS = (".wav", ".ogg", ".mp3", ".flac")

SMOOTH_FRAMES = 8          # nhìn lại bao nhiêu frame gần nhất để bỏ phiếu (chống giật)
MIN_AGREE = 0.6            # loại hợp âm mới phải chiếm ít nhất tỉ lệ này mới được đổi
LOST_FRAMES = 12           # mất tay trái liên tục bấy nhiêu frame thì dừng nhạc
SWAP_HANDS = False         # True nếu app nhận nhầm tay (cũng bấm phím X khi đang chạy)


# ----------------------------------------------------------------------
# PHÁT NHẠC
# ----------------------------------------------------------------------
class SoundPlayer:
    def __init__(self, sounds_dir):
        self.ok, self.muted = False, False
        self.index, self.cache, self.current = {}, {}, None
        try:
            import pygame
            pygame.mixer.init()
            self.pg, self.ok = pygame, True
        except Exception as e:
            print(f"[Cảnh báo] Không bật được âm thanh ({type(e).__name__}: {e}). App vẫn chạy nhưng không có tiếng.")
        if sounds_dir:
            for p in sorted(Path(sounds_dir).rglob("*")):
                if p.suffix.lower() in SOUND_EXTS:
                    self.index.setdefault(p.stem.lower(), p)
            print(f"Thư mục nhạc: {sounds_dir} ({len(self.index)} file)")
        else:
            print("[Cảnh báo] Không tìm thấy thư mục nhạc -> dùng --sounds \"đường\\dẫn\\tới\\sounds\"")

    def find(self, stem):
        return self.index.get(str(stem).lower())

    def play(self, path):
        if not self.ok or self.muted or path is None:
            return
        if self.current == path:
            return
        self.stop()
        if path not in self.cache:
            self.cache[path] = self.pg.mixer.Sound(str(path))
        self.cache[path].play(loops=-1, fade_ms=100)           # giữ tay = phát lặp
        self.current = path

    def stop(self):
        if self.ok and self.current is not None:
            self.cache[self.current].fadeout(300)             # tắt êm
        self.current = None

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.stop()


# ----------------------------------------------------------------------
# LOGIC: dự đoán loại hợp âm, làm mượt, ghép với gốc, phát nhạc
# ----------------------------------------------------------------------
class ChordEngine:
    def __init__(self, info, player, mapping):
        self.model, self.cols = info["model"], info["cols"]
        self.player, self.mapping = player, mapping
        self.root = "C"
        self.history = deque(maxlen=SMOOTH_FRAMES)
        self.lost = 0
        self.raw = None            # dự đoán thô của frame gần nhất
        self.quality = None        # loại hợp âm đã ổn định (đang phát)
        self.conf = 0.0            # tỉ lệ frame gần đây đồng ý với loại hiện tại
        self.missing_file = False

    # --- tên hợp âm ---
    def chord_name(self):
        if self.quality is None:
            return None
        return self.root + QUALITY_SUFFIX.get(self.quality, "?")

    def sound_path(self):
        if self.quality is None:
            return None
        key = f"{self.root}_{self.quality}"
        stem = self.mapping.get(key, self.chord_name())       # mapping.json (nếu có) ưu tiên
        return self.player.find(stem)

    # --- điều khiển ---
    def set_root(self, r):
        self.root = r
        self._sync_sound()

    def _sync_sound(self):
        path = self.sound_path()
        self.missing_file = self.quality is not None and path is None
        if self.quality is None or path is None:       # không có hợp âm hoặc thiếu file -> im lặng
            self.player.stop()
        else:
            self.player.play(path)

    def update(self, left_landmarks):
        """left_landmarks: danh sách 21 điểm của tay trái (MediaPipe) hoặc None nếu không thấy."""
        if left_landmarks is None:
            self.lost += 1
            self.raw = None
            if self.lost >= LOST_FRAMES:
                self.history.clear()
                self.quality, self.conf = None, 0.0
                self._sync_sound()
            return
        self.lost = 0
        feat = common.normalize(left_landmarks, None)              # đúng cách chuẩn hóa lúc train
        self.raw = str(self.model.predict(feat[self.cols].reshape(1, -1))[0])
        self.history.append(self.raw)

        label, cnt = Counter(self.history).most_common(1)[0]
        self.conf = cnt / len(self.history)
        if len(self.history) >= 3 and self.conf >= MIN_AGREE and label != self.quality:
            self.quality = label                                   # chỉ đổi khi đủ ổn định
        elif self.quality is not None:
            self.conf = self.history.count(self.quality) / len(self.history)
        self._sync_sound()


# ----------------------------------------------------------------------
# GIAO DIỆN (chỉ dùng chữ không dấu vì cv2.putText không vẽ được tiếng Việt có dấu)
# ----------------------------------------------------------------------
GREEN, RED, YELLOW, WHITE, GRAY = (80, 220, 80), (70, 70, 240), (60, 210, 240), (245, 245, 245), (150, 150, 150)


def text(img, s, org, scale=0.6, color=WHITE, thick=1):
    """Chữ có bóng đen để dễ đọc trên mọi nền (bóng cùng độ dày nét nên luôn khớp với chữ)."""
    x, y = org
    for dx, dy in ((2, 2), (-1, 1)):
        cv2.putText(img, s, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_ui(img, eng, fps, swapped, muted):
    h, w = img.shape[:2]
    # bảng thông tin góc trên trái (nền mờ)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (360, 215), (25, 25, 25), -1)
    cv2.rectangle(overlay, (0, h - 95), (w, h), (25, 25, 25), -1)           # nền cho hàng nút và dòng hướng dẫn
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    chord = eng.chord_name()
    text(img, chord if chord else "---", (15, 75), 2.2, GREEN if chord else GRAY, 3)

    hand = "PHAI (da doi)" if swapped else "TRAI"
    if eng.lost > 0 and eng.quality is None:
        text(img, f"Khong thay tay {hand.split()[0].lower()}", (15, 110), 0.65, RED)
    elif eng.quality is not None:
        text(img, f"Loai: {QUALITY_DESC.get(eng.quality, eng.quality)}", (15, 110), 0.65, WHITE)
    else:
        text(img, "Dang nhan dien...", (15, 110), 0.65, YELLOW)

    # thanh độ ổn định
    cv2.rectangle(img, (15, 125), (245, 142), (70, 70, 70), -1)
    cv2.rectangle(img, (15, 125), (15 + int(230 * eng.conf), 142), GREEN if eng.conf >= MIN_AGREE else YELLOW, -1)
    text(img, f"on dinh {eng.conf * 100:.0f}%", (255, 140), 0.5)

    text(img, f"Goc: {eng.root}   Tay dieu khien: {hand}", (15, 168), 0.55)
    status = "TAT TIENG" if muted else ("khong co tieng" if not eng.player.ok else "dang phat" if eng.player.current else "im lang")
    text(img, f"Nhac: {status}   FPS: {fps:.0f}", (15, 192), 0.55, YELLOW if muted else WHITE)
    if eng.missing_file:
        text(img, f"Khong co file nhac '{chord}'", (15, 212), 0.5, YELLOW)

    # hàng nút chọn gốc ở dưới
    bw = 46
    x0 = (w - bw * len(ROOTS)) // 2
    for i, r in enumerate(ROOTS):
        sel = r == eng.root
        p1, p2 = (x0 + i * bw, h - 85), (x0 + i * bw + bw - 6, h - 45)
        cv2.rectangle(img, p1, p2, GREEN if sel else (60, 60, 60), -1)
        cv2.rectangle(img, p1, p2, WHITE, 1)
        text(img, r, (p1[0] + 12, p1[1] + 28), 0.8, (0, 0, 0) if sel else WHITE, 2)
    text(img, "A-G: chon goc | X: doi tay | M: tat/bat tieng | ESC: thoat", (15, h - 15), 0.5, GRAY)


# ----------------------------------------------------------------------
def find_sounds_dir(arg):
    cands = ([Path(arg)] if arg else []) + [HERE.parent / "my_dataset" / "sounds", HERE / "sounds",
                                            Path.cwd() / "sounds", Path.cwd() / "my_dataset" / "sounds"]
    return next((c for c in cands if c.is_dir()), None)


def load_mapping():
    """mapping.json (nếu có, dạng {"C_R": "Cm7", ...}) cho phép ghi đè cách đặt tên file nhạc."""
    for p in (HERE / "mapping.json", HERE.parent / "mapping.json"):
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
                    print(f"Dùng {p.name}: {len(data)} mục")
                    return {k: Path(v).stem for k, v in data.items()}
                print(f"[Chú ý] {p.name} có định dạng khác dạng {{nhãn: tên file}} nên bỏ qua")
            except Exception as e:
                print(f"[Chú ý] Không đọc được {p.name}: {e}")
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="số thứ tự webcam (0, 1...) hoặc đường dẫn video")
    ap.add_argument("--sounds", default=None, help="thư mục chứa file nhạc")
    ap.add_argument("--headless", action="store_true", help="không mở cửa sổ (dùng để test)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        sys.exit("Chưa có quality_model.pkl. Chạy trước: python train_quality.py")
    with open(MODEL_PATH, "rb") as f:
        info = pickle.load(f)
    print(f"Model: {info['name']} | balanced accuracy khi train: {info['metrics']['balanced_accuracy']:.3f} "
          f"| {info['metrics']['ms_per_frame']:.1f} ms/frame")

    player = SoundPlayer(find_sounds_dir(args.sounds))
    eng = ChordEngine(info, player, load_mapping())
    for q, suf in QUALITY_SUFFIX.items():
        stem = "C" + suf
        print(f"  loai '{q}' -> file '{stem}': {'có' if player.find(stem) else 'KHÔNG THẤY'}")

    holistic = mp.solutions.holistic.Holistic(model_complexity=0, smooth_landmarks=True,
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5)
    draw = mp.solutions.drawing_utils
    spec = lambda c: draw.DrawingSpec(color=c, thickness=2, circle_radius=2)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Không mở được nguồn video: {args.source}")

    swapped, fps, t_prev, n = SWAP_HANDS, 0.0, time.time(), 0
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = holistic.process(rgb)                  # xử lý ảnh gốc (không lật), như lúc thu dữ liệu

        left, right = res.left_hand_landmarks, res.right_hand_landmarks
        if swapped:
            left, right = right, left
        eng.update(left.landmark if left else None)

        for lms, color in ((left, GREEN), (right, GRAY)):
            if lms is not None:
                draw.draw_landmarks(frame, lms, mp.solutions.holistic.HAND_CONNECTIONS, spec(color), spec(color))

        display = cv2.flip(frame, 1)                 # hiển thị dạng gương cho dễ nhìn
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if n else 1 / max(now - t_prev, 1e-6)
        t_prev, n = now, n + 1
        draw_ui(display, eng, fps, swapped, player.muted)
        last = display

        if args.headless:
            if args.max_frames and n >= args.max_frames:
                break
            continue
        cv2.imshow("Demo hop am bang cu chi tay", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("x"):
            swapped = not swapped
            eng.history.clear()
        elif key == ord("m"):
            player.toggle_mute()
            eng._sync_sound()
        elif 0 < key < 128 and chr(key).upper() in ROOTS:
            eng.set_root(chr(key).upper())

    player.stop()
    cap.release()
    cv2.destroyAllWindows()
    if args.headless and last is not None:
        cv2.imwrite(str(HERE / "demo_frame.png"), last)
        print("Đã lưu demo_frame.png")


if __name__ == "__main__":
    main()