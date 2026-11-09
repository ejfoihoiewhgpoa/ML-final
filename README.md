# Đàn hợp âm bằng cử chỉ tay (Hand Gesture Chord Piano)

Đồ án môn Machine Learning: **Vision-Based Real-Time Hand Landmark Classification for Interactive Audio and Music Applications**.

Chương trình nhận diện hình dạng bàn tay qua webcam, phân loại thành loại hợp âm và phát file nhạc tương ứng theo thời gian thực. Các thuật toán phân loại (**KNN, Decision Tree, Random Forest**) được tự cài đặt từ đầu; MediaPipe chỉ dùng để lấy 21 điểm landmark của bàn tay.

## Cách hoạt động

```
Webcam ──► MediaPipe Holistic ──► 21 landmark tay trái ──► chuẩn hóa (common.py)
                                                              │
                                                              ▼
 Phát file nhạc  ◄── ghép "gốc + loại" ◄── làm mượt (bỏ phiếu 8 frame) ◄── model dự đoán LOẠI hợp âm
                          ▲
                Phím A-G chọn GỐC
```

- **Tay trái** (hình dạng bàn tay) quyết định **loại** hợp âm.
- **Phím A–G** chọn **gốc** hợp âm (gốc không đoán được từ cử chỉ nên chọn bằng phím).
- Hợp âm = gốc + loại. Ví dụ: gốc `C` + loại `maj7` → `Cmaj7` → phát file `Cmaj7`.
- Giữ tay thì nhạc lặp liên tục, hạ tay (mất tay trái khoảng 12 frame) thì nhạc tắt dần.

### Các loại hợp âm model phân loại

| Nhãn dữ liệu | Loại hợp âm | Hậu tố tên file nhạc | Ví dụ |
|:---:|---|:---:|:---:|
| `1` | major | *(không có)* | `C` |
| `D` | dominant 7 | `7` | `C7` |
| `M` | major 7 | `maj7` | `Cmaj7` |
| `m1` | minor | `m` | `Cm` |
| `R` | minor 7 | `m7` | `Cm7` |

> Việc gán nhãn `R` → `m7` là suy đoán từ danh sách file nhạc. Nếu phát sai file, sửa `QUALITY_SUFFIX` trong `main/app.py` hoặc dùng `mapping.json`.

### Cử chỉ tay ứng với từng loại

> *(Điền mô tả hoặc chèn ảnh cho từng cử chỉ. Trong giao diện, mở cửa sổ "Cử chỉ hợp âm" (F2) để chụp ảnh mẫu.)*

| Loại | Cử chỉ tay trái |
|---|---|
| major | |
| dominant 7 | |
| major 7 | |
| minor | |
| minor 7 | |

## Cài đặt

Yêu cầu **Python 3.11**.

```bash
pip install mediapipe==0.10.14 opencv-python pygame numpy pillow
```

> **Quan trọng:** phải dùng `mediapipe==0.10.14`. Các bản mới hơn đã bỏ API `mp.solutions` nên code sẽ báo lỗi `module 'mediapipe' has no attribute 'solutions'`. `pillow` là tùy chọn (giúp video trong giao diện mượt hơn).

Kiểm tra môi trường:

```bash
python -c "import mediapipe as mp, cv2, pygame; h=mp.solutions.hands.Hands(); print('Hands OK', cv2.__version__, pygame.__version__)"
```

## Cách chạy

Chạy từ thư mục gốc của repo:

```bash
# 1. Huấn luyện model phân loại loại hợp âm (một lần) -> tạo main/quality_model.pkl
python main/train_quality.py

# 2a. Chạy app giao diện (khuyên dùng)
python main/chay_app.py

# 2b. Hoặc chạy bản cửa sổ OpenCV đơn giản
python main/app.py
```

Tùy chọn của `app.py`:

| Tham số | Ý nghĩa |
|---|---|
| `--source 1` | dùng webcam khác (mặc định `0`) |
| `--source video.mp4` | chạy trên file video thay vì webcam |
| `--sounds "đường\dẫn\sounds"` | chỉ định thư mục chứa file nhạc |
| `--headless` | không mở cửa sổ (dùng để test) |

Nếu không chỉ định `--sounds`, app tự tìm ở `my_dataset/sounds`, `sounds/` cạnh code, hoặc trong thư mục đang chạy.

### Điều khiển

| Phím | Chức năng |
|:---:|---|
| `A`–`G` | chọn gốc hợp âm |
| `X` | đổi tay (khi app nhận nhầm tay phải thành tay trái) |
| `M` | tắt / bật tiếng |
| `Space` | bắt đầu / dừng *(giao diện)* |
| `F1` | mở hướng dẫn *(giao diện)* |
| `F2` | mở bảng cử chỉ ↔ hợp âm *(giao diện)* |
| `Esc` / `Q` | thoát |

## File nhạc

Đặt các file `.wav` / `.ogg` / `.mp3` / `.flac` trong thư mục `sounds`, đặt tên theo hợp âm: `A.wav`, `A7.wav`, `Am.wav`, `Am7.wav`, `Amaj7.wav`, ... cho cả 7 gốc `A`–`G`.

Muốn đổi cách đặt tên, tạo `mapping.json` với dạng `{"<gốc>_<nhãn>": "<tên file>"}`, ví dụ:

```json
{ "C_R": "Cm7" }
```

## Cấu trúc thư mục

```
progress/
├── main/
│   ├── app.py                # app demo real-time (cửa sổ OpenCV)
│   ├── chay_app.py           # app có giao diện tkinter (video, nút chọn gốc, hướng dẫn)
│   ├── common.py             # chuẩn hóa landmark (dùng chung lúc train và lúc chạy)
│   ├── knn.py                # KNN tự cài đặt
│   ├── decision_tree.py      # Decision Tree tự cài đặt
│   ├── random_forest.py      # Random Forest tự cài đặt
│   ├── train_quality.py      # huấn luyện model loại hợp âm -> quality_model.pkl
│   ├── quality_model.pkl     # model đã huấn luyện
│   ├── mapping.json          # (tùy chọn) ghi đè tên file nhạc
│   ├── compare.py            # so sánh các thuật toán
│   ├── analyze_error.py      # phân tích lỗi dự đoán
│   ├── check_features.py     # kiểm tra đặc trưng
│   ├── plot_*.py             # vẽ biểu đồ (cây quyết định, bàn tay, kết quả)
│   ├── ket_qua.csv           # kết quả thí nghiệm
│   ├── melody.py             # thử nghiệm phát giai điệu
│   ├── thu_nghiem_khac.py    # các thí nghiệm khác
│   ├── 21_points.ipynb       # notebook khảo sát 21 điểm landmark
│   ├── chord.ipynb           # notebook thử nghiệm hợp âm
│   └── figs/                 # hình và file kết quả
├── check_camera/             # kiểm tra webcam
├── my_dataset/               # dữ liệu và thư mục nhạc (sounds/)
└── README.md
```

## Ghi chú

- Dữ liệu chia theo **nhóm** (train/test không trùng nhóm) để đánh giá đúng khả năng tổng quát hóa.
- Các file lớn (`landmarks_cache.npz`, một số `.pkl`) có thể không được đưa lên GitHub do giới hạn dung lượng; chạy lại script huấn luyện để tạo lại.
- Khung video trong app hiển thị dạng gương, nhưng nhận diện chạy trên ảnh gốc (không lật) để khớp với lúc thu dữ liệu.