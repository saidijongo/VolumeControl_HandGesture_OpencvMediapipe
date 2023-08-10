import cv2
import numpy as np
import math
import time
import pyautogui
from pynput.keyboard import Key, Controller
keyboard = Controller()

################################
wCam, hCam = 1280, 720
################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

minHand = 50  # 50
maxHand = 300  # 300

yellow_lower = np.array([22, 93, 0])
yellow_upper = np.array([45, 255, 255])
prev_y = 0

def draw_volume_bar(img, length):
    vol_bar = int(np.interp(length, [minHand, maxHand], [400, 150]))
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, vol_bar), (85, 400), (255, 0, 0), cv2.FILLED)
    vol_percentage = int(np.interp(length, [minHand, maxHand], [0, 100]))
    cv2.putText(img, f'{vol_percentage}%', (40, 430), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 255, 255), 2)

import mediapipe as mp

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

last_length = None
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []  # Initialize the list
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(lmList) >= 9:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

                length = math.hypot(x2 - x1, y2 - y1)

                if last_length:
                    if length > last_length:
                        keyboard.press(Key.media_volume_up)
                        keyboard.release(Key.media_volume_up)
                        print("VOL UP")
                    elif length < last_length:
                        keyboard.press(Key.media_volume_down)
                        keyboard.release(Key.media_volume_down)
                        print("VOL DOWN")

                last_length = length

                draw_volume_bar(img, length)

    cv2.imshow('frame', img)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
