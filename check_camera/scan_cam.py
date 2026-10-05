import cv2, time

for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
    for i in range(4):
        cap = cv2.VideoCapture(i, backend)
        if not cap.isOpened():
            print(backend_name, i, "không mở được")
            continue
        time.sleep(1)                      # chờ camera khởi động
        mean = 0
        for _ in range(30):                # bỏ qua các frame đầu
            ok, img = cap.read()
            if ok:
                mean = img.mean()
        print(backend_name, i, "độ sáng trung bình:", round(float(mean), 1))
        cap.release()