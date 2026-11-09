"""
GIAO DIỆN (tkinter) cho app đàn hợp âm bằng cử chỉ tay.

Cách dùng:
  1. Lưu file này cạnh app.py (cùng thư mục với quality_model.pkl, common.py ...).
  2. Chạy:  python gui_app.py
  3. Chọn nguồn (webcam 0/1... hoặc file video), thư mục nhạc, bấm "Bắt đầu".

Phím tắt: A-G chọn gốc | X đổi tay | M tắt/bật tiếng | Space bắt đầu/dừng | F1 hướng dẫn | Esc thoát
"""
import base64
import pickle
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import mediapipe as mp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import app as core  # noqa: E402  (dùng lại SoundPlayer, ChordEngine, cấu hình từ app.py)

try:
    from PIL import Image, ImageTk
except ImportError:  # không có Pillow thì dùng PNG qua tk.PhotoImage
    Image = ImageTk = None

BG, PANEL, FG, GREEN, YELLOW, RED, GRAY = "#1e1e1e", "#2b2b2b", "#f0f0f0", "#50dc50", "#f0d23c", "#f04646", "#969696"
VIDEO_W = 720

# ---- BẢNG CỬ CHỈ ----------------------------------------------------------
# Thứ tự hiển thị trong cửa sổ "Cử chỉ hợp âm" (nhãn dữ liệu của model).
GUIDE_ORDER = ["1", "D", "M", "m1", "R"]
# Mô tả bằng chữ cho từng cử chỉ: BẠN TỰ ĐIỀN theo đúng hình tay lúc thu dữ liệu,
# ví dụ "1": "Nắm tay, chỉ duỗi ngón trỏ".  Để trống thì app hiện lời nhắc chụp mẫu.
GESTURE_TEXT = {
    "1": "",
    "D": "",
    "M": "",
    "m1": "",
    "R": "",
}
# Ảnh mẫu (chụp từ camera bằng nút "Chụp mẫu") được lưu ở đây và tự nạp lại lần sau.
SAMPLE_DIR = HERE / "gesture_samples"
THUMB = 120


def to_photo(bgr):
    """Chuyển ảnh OpenCV (BGR) thành ảnh hiển thị được trong tkinter."""
    if Image is not None:
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    ok, buf = cv2.imencode(".png", bgr)
    return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))

QUICK_START = (
    "HƯỚNG DẪN NHANH\n\n"
    "1. Bấm  ▶ Bắt đầu  (hoặc phím Space)\n"
    "2. Đưa TAY TRÁI vào khung hình camera\n"
    "3. Đổi hình dạng bàn tay để chọn LOẠI hợp âm\n"
    "4. Bấm phím A-G (hoặc nút bên phải) để chọn GỐC\n"
    "5. Giữ tay = nhạc vang liên tục, hạ tay = dừng\n\n"
    "Bấm F2 / nút  🖐 Cử chỉ hợp âm  để xem giơ tay thế nào ra hợp âm nào\n"
    "Bấm F1 / nút  ❓ Hướng dẫn  để xem hướng dẫn chi tiết"
)

HELP_TEXT = """CÁCH DÙNG

1. Chuẩn bị (làm một lần)
   • Chạy train_quality.py để có file quality_model.pkl.
   • Chuẩn bị thư mục chứa file nhạc (A.wav, Am.wav, A7.wav, Am7.wav, Amaj7.wav ...).

2. Bắt đầu
   • Ô "Nguồn video": nhập 0 (webcam mặc định), 1 (webcam khác) hoặc bấm "..." để chọn file video.
   • Ô "Thư mục nhạc": chọn thư mục chứa file nhạc, hoặc để trống để app tự tìm.
   • Bấm "Bắt đầu" (hoặc phím Space). Bấm lại để dừng.

3. Chơi nhạc
   • TAY TRÁI (hình dạng bàn tay) quyết định LOẠI hợp âm:
       major, dominant 7, major 7, minor, minor 7.
   • GỐC hợp âm chọn bằng phím A-G hoặc bấm nút A-G ở bên phải
     (gốc không đoán được từ cử chỉ nên phải chọn thủ công).
   • Xem cửa sổ "🖐 Cử chỉ hợp âm" (F2) để biết giơ tay thế nào ra loại nào.
   • Hợp âm = gốc + loại. Ví dụ gốc C + loại major 7 → Cmaj7 → phát file Cmaj7.
   • Giữ tay thì nhạc lặp liên tục. Hạ tay (mất tay trái một lúc) thì nhạc tắt dần.

4. Các nút và phím tắt
   • A-G ........ chọn gốc hợp âm
   • X .......... đổi tay (dùng khi app nhận nhầm tay phải thành tay trái)
   • M .......... tắt / bật tiếng
   • Space ...... bắt đầu / dừng
   • F1 ......... mở hướng dẫn này
   • F2 ......... mở bảng cử chỉ tay ↔ hợp âm
   • Esc ........ thoát
   (Khi đang gõ trong ô nhập thì phím tắt tạm không có tác dụng.)

5. Đọc các chỉ số trên màn hình
   • Tên hợp âm lớn: hợp âm đang phát (--- nghĩa là chưa có).
   • Thanh "ổn định": tỉ lệ các frame gần đây cùng đồng ý với loại hợp âm hiện tại.
     Hợp âm chỉ đổi khi đủ ổn định, nên đổi hình tay xong hãy giữ yên khoảng nửa giây.
   • "Nhạc": đang phát / im lặng / TẮT TIẾNG.

MẸO ĐỂ NHẬN DIỆN TỐT
   • Ánh sáng đủ, nền phía sau không quá rối, bàn tay nằm trọn trong khung hình.
   • Đặt tay cách camera khoảng một sải tay, không che ngón tay.
   • Khung video hiển thị dạng gương, nhưng nhận diện chạy trên ảnh gốc nên đừng lo tay trái bị đảo.

XỬ LÝ SỰ CỐ
   • "Không thấy tay trái": đưa tay vào khung hình, hoặc thử bấm X nếu app đang bắt tay phải.
   • "Không có file nhạc '...'": tên file nhạc không khớp tên hợp âm. Kiểm tra thư mục nhạc
     và tên file, hoặc tạo mapping.json để ghi đè (dạng {"C_R": "Cm7"}).
   • Hợp âm "minor 7" đang là suy đoán về nhãn R trong dữ liệu; nếu phát sai file thì sửa
     QUALITY_SUFFIX trong app.py.
   • Không mở được camera: thử đổi nguồn sang 1, hoặc kiểm tra quyền camera trong Windows.
   • Không có tiếng: kiểm tra có đang tắt tiếng (M) không, và loa/tai nghe của máy.
"""


class GuiApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Đàn hợp âm bằng cử chỉ tay")
        self.configure(bg=BG)
        self.minsize(1000, 560)

        self.info = None
        self.cap = self.holistic = self.player = self.eng = None
        self.running = False
        self.fps, self.t_prev, self.n = 0.0, time.time(), 0
        self._img_ref = None
        self.help_win = None
        self.guide_win = None
        self.guide_rows, self.guide_imgs = {}, {}
        self.guide_status = None
        self.countdown_job = None
        self.last_crop, self.last_crop_t = None, 0.0
        self.draw = mp.solutions.drawing_utils

        self.source_var = tk.StringVar(value="0")
        self.sounds_var = tk.StringVar(value="")
        self.root_var = tk.StringVar(value="C")
        self.swap_var = tk.BooleanVar(value=core.SWAP_HANDS)
        self.mute_var = tk.BooleanVar(value=False)

        self._build()
        self.bind("<Key>", self.on_key)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI
    def _build(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Green.Horizontal.TProgressbar", troughcolor="#464646", background=GREEN)

        left = tk.Frame(self, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)
        self.video = tk.Label(left, bg="black", fg=GRAY, text=QUICK_START, justify="left",
                              font=("Segoe UI", 13), width=60, height=22)
        self.video.pack(fill="both", expand=True)

        right = tk.Frame(self, bg=PANEL, width=300)
        right.pack(side="right", fill="y", padx=(5, 10), pady=10)
        right.pack_propagate(False)

        def lbl(parent, text, **kw):
            return tk.Label(parent, text=text, bg=PANEL, fg=kw.pop("fg", FG), **kw)

        # --- nguồn video
        lbl(right, "Nguồn video (số webcam hoặc file)", anchor="w").pack(fill="x", padx=12, pady=(12, 2))
        row = tk.Frame(right, bg=PANEL)
        row.pack(fill="x", padx=12)
        self.source_entry = tk.Entry(row, textvariable=self.source_var)
        self.source_entry.pack(side="left", fill="x", expand=True)
        tk.Button(row, text="...", width=3, command=self.browse_video).pack(side="left", padx=(4, 0))

        # --- thư mục nhạc
        lbl(right, "Thư mục nhạc (để trống = tự tìm)", anchor="w").pack(fill="x", padx=12, pady=(10, 2))
        row = tk.Frame(right, bg=PANEL)
        row.pack(fill="x", padx=12)
        self.sounds_entry = tk.Entry(row, textvariable=self.sounds_var)
        self.sounds_entry.pack(side="left", fill="x", expand=True)
        tk.Button(row, text="...", width=3, command=self.browse_sounds).pack(side="left", padx=(4, 0))

        self.btn_start = tk.Button(right, text="▶  Bắt đầu", font=("Segoe UI", 12, "bold"),
                                   bg="#2e7d32", fg="white", activebackground="#388e3c",
                                   command=self.toggle_run)
        self.btn_start.pack(fill="x", padx=12, pady=12, ipady=4)

        # --- hợp âm hiện tại
        self.chord_lbl = lbl(right, "---", fg=GRAY, font=("Segoe UI", 44, "bold"))
        self.chord_lbl.pack(pady=(0, 0))
        self.quality_lbl = lbl(right, "Chưa chạy", font=("Segoe UI", 11))
        self.quality_lbl.pack()
        self.bar = ttk.Progressbar(right, style="Green.Horizontal.TProgressbar", maximum=100)
        self.bar.pack(fill="x", padx=12, pady=(8, 0))
        self.conf_lbl = lbl(right, "ổn định 0%", fg=GRAY, font=("Segoe UI", 9))
        self.conf_lbl.pack()

        # --- chọn gốc
        lbl(right, "Gốc hợp âm (phím A-G)", anchor="w").pack(fill="x", padx=12, pady=(12, 2))
        roots = tk.Frame(right, bg=PANEL)
        roots.pack(padx=12)
        for r in core.ROOTS:
            tk.Radiobutton(roots, text=r, value=r, variable=self.root_var, indicatoron=False,
                           width=3, pady=6, font=("Segoe UI", 12, "bold"), bg="#3c3c3c", fg=FG,
                           selectcolor=GREEN, activebackground="#555555",
                           command=self.on_root).pack(side="left", padx=1)

        # --- tuỳ chọn
        tk.Checkbutton(right, text="Đổi tay (X)", variable=self.swap_var, command=self.on_swap,
                       bg=PANEL, fg=FG, selectcolor="#3c3c3c", activebackground=PANEL,
                       activeforeground=FG).pack(anchor="w", padx=12, pady=(12, 0))
        tk.Checkbutton(right, text="Tắt tiếng (M)", variable=self.mute_var, command=self.on_mute,
                       bg=PANEL, fg=FG, selectcolor="#3c3c3c", activebackground=PANEL,
                       activeforeground=FG).pack(anchor="w", padx=12)

        # --- trạng thái
        self.music_lbl = lbl(right, "Nhạc: -", anchor="w")
        self.music_lbl.pack(fill="x", padx=12, pady=(12, 0))
        self.warn_lbl = lbl(right, "", anchor="w", fg=YELLOW, wraplength=270, justify="left")
        self.warn_lbl.pack(fill="x", padx=12)
        self.fps_lbl = lbl(right, "FPS: -", anchor="w", fg=GRAY)
        self.fps_lbl.pack(fill="x", padx=12, side="bottom", pady=(0, 10))
        tk.Button(right, text="❓  Hướng dẫn (F1)", bg="#3c3c3c", fg=FG, activebackground="#555555",
                  activeforeground=FG, command=self.open_help).pack(fill="x", padx=12, side="bottom", pady=(0, 6))
        tk.Button(right, text="🖐  Cử chỉ hợp âm (F2)", bg="#3c3c3c", fg=FG, activebackground="#555555",
                  activeforeground=FG, command=self.open_guide).pack(fill="x", padx=12, side="bottom", pady=(0, 6))

    # ------------------------------------------------------------- sự kiện
    def browse_video(self):
        p = filedialog.askopenfilename(title="Chọn file video",
                                       filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("Tất cả", "*.*")])
        if p:
            self.source_var.set(p)

    def browse_sounds(self):
        p = filedialog.askdirectory(title="Chọn thư mục chứa file nhạc")
        if p:
            self.sounds_var.set(p)

    def open_help(self):
        if self.help_win is not None and self.help_win.winfo_exists():
            self.help_win.lift()
            self.help_win.focus_force()
            return
        win = self.help_win = tk.Toplevel(self)
        win.title("Hướng dẫn sử dụng")
        win.configure(bg=BG)
        win.geometry("640x620")
        box = tk.Frame(win, bg=BG)
        box.pack(fill="both", expand=True, padx=10, pady=10)
        sb = tk.Scrollbar(box)
        sb.pack(side="right", fill="y")
        txt = tk.Text(box, wrap="word", bg=PANEL, fg=FG, font=("Segoe UI", 11), relief="flat",
                      padx=12, pady=10, yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.config(command=txt.yview)
        txt.insert("1.0", HELP_TEXT)
        txt.config(state="disabled")
        win.bind("<Escape>", lambda e: win.destroy())

    # ------------------------------------------------- bảng cử chỉ tay
    def grab_hand_crop(self, frame, lm):
        """Lưu ảnh cắt vuông quanh bàn tay trái (dạng gương, giống cái người dùng nhìn thấy)."""
        h, w = frame.shape[:2]
        xs, ys = [p.x * w for p in lm], [p.y * h for p in lm]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.65 + 10
        x0, x1 = int(max(cx - half, 0)), int(min(cx + half, w))
        y0, y1 = int(max(cy - half, 0)), int(min(cy + half, h))
        if x1 - x0 < 20 or y1 - y0 < 20:
            return
        self.last_crop = cv2.flip(frame[y0:y1, x0:x1].copy(), 1)
        self.last_crop_t = time.time()

    def set_guide_status(self, msg, color=FG):
        if self.guide_status is not None and self.guide_win is not None and self.guide_win.winfo_exists():
            self.guide_status.config(text=msg, fg=color)

    def load_thumb(self, q):
        path = SAMPLE_DIR / f"{q}.png"
        row = self.guide_rows.get(q)
        if row is None or not path.exists():
            return
        img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        photo = to_photo(cv2.resize(img, (THUMB, THUMB)))
        self.guide_imgs[q] = photo                       # giữ tham chiếu để ảnh không bị xoá
        row[1].config(image=photo, text="", width=THUMB, height=THUMB)

    def open_guide(self):
        if self.guide_win is not None and self.guide_win.winfo_exists():
            self.guide_win.lift()
            self.guide_win.focus_force()
            return
        win = self.guide_win = tk.Toplevel(self)
        win.title("Cử chỉ tay ↔ hợp âm")
        win.configure(bg=BG)
        win.geometry("620x780")
        tk.Label(win, bg=BG, fg=FG, justify="left", anchor="w", wraplength=580, font=("Segoe UI", 10),
                 text="Dùng TAY TRÁI. Mỗi hình tay ứng với một LOẠI hợp âm; gốc (A-G) chọn bằng phím.\n"
                      "Khi app đang chạy, hàng nào sáng viền xanh là loại app đang nhận ra từ tay bạn.\n"
                      "Muốn lưu ảnh mẫu: bấm Bắt đầu, bấm \"Chụp mẫu\" ở hàng muốn gán, rồi làm đúng "
                      "cử chỉ đó trước camera trong 3 giây."
                 ).pack(fill="x", padx=12, pady=(10, 4))
        self.guide_status = tk.Label(win, bg=BG, fg=FG, anchor="w", text="", font=("Segoe UI", 10, "bold"))
        self.guide_status.pack(fill="x", padx=12)

        self.guide_rows, self.guide_imgs = {}, {}
        for q in GUIDE_ORDER:
            row = tk.Frame(win, bg=PANEL, highlightthickness=3, highlightbackground=PANEL)
            row.pack(fill="x", padx=12, pady=4)
            thumb = tk.Label(row, bg="black", fg=GRAY, text="chưa có\nảnh mẫu", width=14, height=7)
            thumb.pack(side="left", padx=8, pady=8)
            info = tk.Frame(row, bg=PANEL)
            info.pack(side="left", fill="both", expand=True, padx=(0, 8))
            desc = core.QUALITY_DESC.get(q, q)
            ex = "C" + core.QUALITY_SUFFIX.get(q, "?")
            tk.Label(info, text=f"{desc}   (ví dụ: {ex})", bg=PANEL, fg=FG, anchor="w",
                     font=("Segoe UI", 12, "bold")).pack(fill="x", pady=(8, 2))
            tk.Label(info, text=GESTURE_TEXT.get(q) or "Chưa có mô tả cử chỉ cho loại này.",
                     bg=PANEL, fg=GRAY, anchor="w", justify="left", wraplength=360).pack(fill="x")
            tk.Button(info, text="📷 Chụp mẫu (sau 3 giây)",
                      command=lambda q=q: self.capture_sample(q)).pack(anchor="w", pady=8)
            self.guide_rows[q] = (row, thumb)
            self.load_thumb(q)
        win.bind("<Escape>", lambda e: win.destroy())

    def capture_sample(self, q):
        if not self.running:
            self.set_guide_status("Hãy bấm ▶ Bắt đầu ở cửa sổ chính trước, rồi chụp mẫu.", YELLOW)
            return
        if self.countdown_job is not None:
            self.after_cancel(self.countdown_job)
        self.countdown(q, 3)

    def countdown(self, q, n):
        self.countdown_job = None
        if not self.running:
            return
        label = core.QUALITY_DESC.get(q, q)
        if n > 0:
            self.set_guide_status(f"Làm cử chỉ '{label}' bằng tay trái... chụp sau {n}", YELLOW)
            self.countdown_job = self.after(1000, self.countdown, q, n - 1)
            return
        if self.last_crop is None or time.time() - self.last_crop_t > 1.0:
            self.set_guide_status("Không thấy tay trái trong khung hình, thử lại.", RED)
            return
        SAMPLE_DIR.mkdir(exist_ok=True)
        ok, buf = cv2.imencode(".png", cv2.resize(self.last_crop, (240, 240)))
        if ok:
            (SAMPLE_DIR / f"{q}.png").write_bytes(buf.tobytes())
            self.load_thumb(q)
            self.set_guide_status(f"Đã lưu ảnh mẫu cho '{label}'.", GREEN)

    def on_key(self, e):
        if isinstance(self.focus_get(), tk.Entry):      # đang gõ trong ô nhập thì không bắt phím tắt
            return
        k = e.keysym
        if k == "F1":
            self.open_help()
        elif k == "F2":
            self.open_guide()
        elif k == "Escape":
            self.on_close()
        elif k == "space":
            self.toggle_run()
        elif k.lower() == "x":
            self.swap_var.set(not self.swap_var.get())
            self.on_swap()
        elif k.lower() == "m":
            self.mute_var.set(not self.mute_var.get())
            self.on_mute()
        elif len(k) == 1 and k.upper() in core.ROOTS:
            self.root_var.set(k.upper())
            self.on_root()

    def on_root(self):
        if self.eng:
            self.eng.set_root(self.root_var.get())

    def on_swap(self):
        if self.eng:
            self.eng.history.clear()

    def on_mute(self):
        if self.player and self.player.muted != self.mute_var.get():
            self.player.toggle_mute()
            self.eng._sync_sound()

    def toggle_run(self):
        self.stop() if self.running else self.start()

    # --------------------------------------------------------- chạy / dừng
    def start(self):
        if self.info is None:
            if not core.MODEL_PATH.exists():
                messagebox.showerror("Thiếu model", "Chưa có quality_model.pkl.\nChạy trước: python train_quality.py")
                return
            with open(core.MODEL_PATH, "rb") as f:
                self.info = pickle.load(f)

        src = self.source_var.get().strip().strip('"')
        source = int(src) if src.isdigit() else src
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            messagebox.showerror("Lỗi nguồn video", f"Không mở được: {src}\nThử đổi số webcam (0, 1...).")
            return
        self.cap = cap

        self.player = core.SoundPlayer(core.find_sounds_dir(self.sounds_var.get().strip().strip('"') or None))
        self.eng = core.ChordEngine(self.info, self.player, core.load_mapping())
        self.eng.root = self.root_var.get()
        self.mute_var.set(False)
        self.holistic = mp.solutions.holistic.Holistic(model_complexity=0, smooth_landmarks=True,
                                                       min_detection_confidence=0.5,
                                                       min_tracking_confidence=0.5)
        self.running = True
        self.fps, self.t_prev, self.n = 0.0, time.time(), 0
        self.btn_start.config(text="■  Dừng", bg="#c62828", activebackground="#d32f2f")
        self.source_entry.config(state="disabled")
        self.sounds_entry.config(state="disabled")
        self.loop()

    def stop(self):
        self.running = False
        if self.player:
            self.player.stop()
        if self.cap:
            self.cap.release()
        if self.holistic:
            self.holistic.close()
        self.cap = self.holistic = self.player = self.eng = None
        self.btn_start.config(text="▶  Bắt đầu", bg="#2e7d32", activebackground="#388e3c")
        self.source_entry.config(state="normal")
        self.sounds_entry.config(state="normal")
        self.video.config(image="", text=QUICK_START, width=60, height=22)
        self._img_ref = None
        self.chord_lbl.config(text="---", fg=GRAY)
        self.quality_lbl.config(text="Chưa chạy", fg=FG)
        self.bar["value"] = 0
        self.conf_lbl.config(text="ổn định 0%")
        self.music_lbl.config(text="Nhạc: -")
        self.warn_lbl.config(text="")
        self.fps_lbl.config(text="FPS: -")

    def on_close(self):
        self.running = False
        if self.player:
            self.player.stop()
        if self.cap:
            self.cap.release()
        self.destroy()

    # ------------------------------------------------------------- vòng lặp
    def loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if not ok:                                   # hết video hoặc mất camera
            self.stop()
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.holistic.process(rgb)             # xử lý ảnh gốc (không lật), như lúc thu dữ liệu
        left, right = res.left_hand_landmarks, res.right_hand_landmarks
        if self.swap_var.get():
            left, right = right, left
        self.eng.update(left.landmark if left else None)

        if left is not None:
            self.grab_hand_crop(frame, left.landmark)
        spec = lambda c: self.draw.DrawingSpec(color=c, thickness=2, circle_radius=2)
        for lms, color in ((left, core.GREEN), (right, core.GRAY)):
            if lms is not None:
                self.draw.draw_landmarks(frame, lms, mp.solutions.holistic.HAND_CONNECTIONS, spec(color), spec(color))

        self.show_frame(cv2.flip(frame, 1))
        self.update_panel()

        now = time.time()
        inst = 1 / max(now - self.t_prev, 1e-6)
        self.fps = 0.9 * self.fps + 0.1 * inst if self.n else inst
        self.t_prev, self.n = now, self.n + 1
        self.fps_lbl.config(text=f"FPS: {self.fps:.0f}")
        self.after(1, self.loop)

    def show_frame(self, bgr):
        h, w = bgr.shape[:2]
        bgr = cv2.resize(bgr, (VIDEO_W, int(h * VIDEO_W / w)))
        img = to_photo(bgr)
        self.video.config(image=img, text="", width=bgr.shape[1], height=bgr.shape[0])
        self._img_ref = img                          # giữ tham chiếu để ảnh không bị xoá

    def update_panel(self):
        eng, player = self.eng, self.player
        chord = eng.chord_name()
        self.chord_lbl.config(text=chord if chord else "---", fg=GREEN if chord else GRAY)

        if eng.lost > 0 and eng.quality is None:
            hand = "phải" if self.swap_var.get() else "trái"
            self.quality_lbl.config(text=f"Không thấy tay {hand}", fg=RED)
        elif eng.quality is not None:
            self.quality_lbl.config(text=f"Loại: {core.QUALITY_DESC.get(eng.quality, eng.quality)}", fg=FG)
        else:
            self.quality_lbl.config(text="Đang nhận diện...", fg=YELLOW)

        self.bar["value"] = eng.conf * 100
        self.conf_lbl.config(text=f"ổn định {eng.conf * 100:.0f}%")

        if player.muted:
            status = "TẮT TIẾNG"
        elif not player.ok:
            status = "không có tiếng (lỗi pygame)"
        else:
            status = "đang phát" if player.current else "im lặng"
        self.music_lbl.config(text=f"Nhạc: {status}", fg=YELLOW if player.muted else FG)
        self.warn_lbl.config(text=f"Không có file nhạc '{chord}'" if eng.missing_file else "")
        if self.guide_win is not None and self.guide_win.winfo_exists():
            for q, (row, _) in self.guide_rows.items():
                row.config(highlightbackground=GREEN if q == eng.quality else PANEL)


if __name__ == "__main__":
    GuiApp().mainloop()