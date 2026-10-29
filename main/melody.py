import os, cv2, pygame, mediapipe as mp

pygame.mixer.init()

FILES = {
    1: "sounds/C.wav",
    2: "sounds/Am.wav",
    3: "sounds/F.wav",
    4: "sounds/G.wav",
}
NAMES = {1: "C", 2: "Am", 3: "F", 4: "G"}   # chữ hiện trên màn hình

sounds = {k: pygame.mixer.Sound(p) for k, p in FILES.items()}

def count_fingers(lm):
    tips, pips = [8, 12, 16, 20], [6, 10, 14, 18]
    return sum(lm[t].y < lm[p].y for t, p in zip(tips, pips))

hands = mp.solutions.hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

current = 0                 # nhãn đang phát (0 = không phát)
candidate, stable = 0, 0

while True:
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    n = count_fingers(res.multi_hand_landmarks[0].landmark) if res.multi_hand_landmarks else 0
    if n not in sounds:
        n = 0

    # chỉ đổi trạng thái khi cử chỉ ổn định vài khung hình
    stable = stable + 1 if n == candidate else 0
    candidate = n

    if stable == 4 and n != current:
        if current:
            sounds[current].fadeout(300)           # tắt bài cũ êm
        if n:
            sounds[n].play(loops=-1, fade_ms=100)  # giữ tay = phát mãi
        current = n

    if current:
        cv2.putText(frame, NAMES[current], (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow("Phat nhac bang tay", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

pygame.mixer.stop()
cap.release(); cv2.destroyAllWindows()