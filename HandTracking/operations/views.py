from django.shortcuts import render
import cv2
import numpy as np
from django.shortcuts import redirect
from .HandTrackingModule import *
import time
import autopy


import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


from .keys import *
from .handTracker import *
from pynput.keyboard import Controller


# Create your views here.


def mousetrack(request):
    ##########################
    wCam, hCam = 640, 480
    frameR = 100  # Frame Reduction
    smoothening = 10
    #########################

    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    # detector = htm.handDetector(maxHands=1)
    detector = handDetector(maxHands=1)
    wScr, hScr = autopy.screen.size()
    # print(wScr, hScr)

    while True:
        # 1. Find hand Landmarks
        success, img = cap.read()
        print(img,success)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        # 2. Get the tip of the index and middle fingers
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            # print(x1, y1, x2, y2)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)
        # 4. Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            # 6. Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7. Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # 8. Both Index and middle fingers are up : Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 10. Click mouse if distance short
            if length < 30:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

        # 11. Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        # 12. Display
        cv2.imshow("Image", img)
        key = cv2.waitKey(33)
        if key != -1:
            print(key)

        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')


def indexView(request):
    return render(request, 'index.html')

def volumecontrol(request):
    wCam, hCam = 640, 480
    ################################

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0

    detector = handDetector(detectionCon=0.7, maxHands=1)

    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # volume.GetMute()
    # volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    vol = 0
    volBar = 400
    volPer = 0
    area = 0
    colorVol = (255, 0, 0)

    while True:
        success, img = cap.read()
        print(img,success)
        # Find Hand
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img, draw=True)
        if len(lmList) != 0:

            # Filter based on size
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100
            # print(area)
            if 250 < area < 1000:

                # Find Distance between index and Thumb
                length, img, lineInfo = detector.findDistance(4, 8, img)
                # print(length)

                # Convert Volume
                volBar = np.interp(length, [50, 200], [400, 150])
                volPer = np.interp(length, [50, 200], [0, 100])

                # Reduce Resolution to make it smoother
                smoothness = 10
                volPer = smoothness * round(volPer / smoothness)

                # Check fingers up
                fingers = detector.fingersUp()
                # print(fingers)

                # If pinky is down set volume
                if not fingers[4]:
                    volume.SetMasterVolumeLevelScalar(volPer / 100, None)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    colorVol = (0, 255, 0)
                else:
                    colorVol = (255, 0, 0)

        # Drawings
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)
        cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
        cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, colorVol, 3)

        # Frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 3)

        cv2.imshow("Img", img)
        cv2.waitKey(1)
        key = cv2.waitKey(33)
        if key != -1:
            print(key)

        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')

def getMousPos(event, x, y, flags, param):
    global clickedX, clickedY

    if event == cv2.EVENT_LBUTTONUP:

        clickedX, clickedY = x, y
    if event == cv2.EVENT_MOUSEMOVE:
        pass

def calculateIntDidtance(pt1, pt2):
    return int(((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5)

def keyboardcontrol(request):
    print("req1", request)
    w, h = 80, 60
    startX, startY = 40, 200
    keys = []
    letters = list("QWERTYUIOPASDFGHJKLZXCVBNM")
    for i, l in enumerate(letters):
        if i < 10:
            keys.append(Key(startX + i * w + i * 5, startY, w, h, l))
        elif i < 19:
            keys.append(Key(startX + (i - 10) * w + i * 5, startY + h + 5, w, h, l))
        else:
            keys.append(Key(startX + (i - 19) * w + i * 5, startY + 2 * h + 10, w, h, l))

    keys.append(Key(startX + 25, startY + 3 * h + 15, 5 * w, h, "Space"))
    keys.append(Key(startX + 8 * w + 50, startY + 2 * h + 10, w, h, "clr"))
    keys.append(Key(startX + 5 * w + 30, startY + 3 * h + 15, 5 * w, h, "<--"))

    textBox = Key(startX, startY - h - 5, 10 * w + 9 * 5, h, '')

    cap = cv2.VideoCapture(0)
    ptime = 0

    # initiating the hand tracker
    tracker = HandTracker(detectionCon=0.8)

    # getting frame's height and width
    frameHeight, frameWidth, _ = cap.read()[1].shape

    clickedX, clickedY = 0, 0
    mousX, mousY = 0, 0

    show = False
    cv2.namedWindow('video')
    counter = 0
    previousClick = 0

    keyboard = Controller()
    while True:
        if counter > 0:
            counter -= 1

        signTipX = 0
        signTipY = 0

        thumbTipX = 0
        thumbTipY = 0

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (int(frameWidth * 1.5), int(frameHeight * 1.5)))
        frame = cv2.flip(frame, 1)
        # find hands
        frame = tracker.findHands(frame)
        lmList = tracker.getPostion(frame, draw=False)
        if lmList:
            signTipX, signTipY = lmList[8][1], lmList[8][2]
            thumbTipX, thumbTipY = lmList[12][1], lmList[12][2]
            if calculateIntDidtance((signTipX, signTipY), (thumbTipX, thumbTipY)) < 50:
                centerX = int((signTipX + thumbTipX) / 2)
                centerY = int((signTipY + thumbTipY) / 2)
                cv2.line(frame, (signTipX, signTipY), (thumbTipX, thumbTipY), (0, 255, 0), 2)
                cv2.circle(frame, (centerX, centerY), 5, (0, 255, 0), cv2.FILLED)

        ctime = time.time()
        fps = int(1 / (ctime - ptime))

        cv2.putText(frame, str(fps) + " FPS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.setMouseCallback('video', getMousPos)

        # checking if sign finger is over a key and if click happens
        alpha = 0.5
        # if show:
        textBox.drawKey(frame, (255, 255, 255), (0, 0, 0), 0.3)
        # k.isOver(mouseX, mouseY) or
        for k in keys:
            if k.isOver(signTipX, signTipY):
                alpha = 0.1
                # writing using mouse right click
                if k.isOver(clickedX, clickedY):
                    if k.text == '<--':
                        textBox.text = textBox.text[:-1]
                    elif k.text == 'clr':
                        textBox.text = ''
                    elif len(textBox.text) < 30:
                        if k.text == 'Space':
                            textBox.text += " "
                        else:
                            textBox.text += k.text

                # writing using fingers
                if (k.isOver(thumbTipX, thumbTipY)):
                    clickTime = time.time()
                    if clickTime - previousClick > 0.4:
                        if k.text == '<--':
                            textBox.text = textBox.text[:-1]
                        elif k.text == 'clr':
                            textBox.text = ''
                        elif len(textBox.text) < 20:
                            if k.text == 'Space':
                                textBox.text += " "
                            else:
                                textBox.text += k.text
                                # simulating the press of actuall keyboard
                                keyboard.press(k.text)
                        previousClick = clickTime
            k.drawKey(frame, (255, 255, 255), (0, 0, 0), alpha=alpha)
            alpha = 0.5
        clickedX, clickedY = 0, 0
        ptime = ctime
        cv2.imshow('video', frame)

        # stop the video when 'q' is pressed
        pressedKey = cv2.waitKey(1)
        if pressedKey == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    return redirect('/')


def zoomcontrol(request):
    print("shbfyv", request)
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = handDetector(detectionCon=0.7, maxHands=2)
    startDist = None
    scale = 0
    cx, cy = 500, 500
    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)

        img1 = cv2.imread("sinhgad.jpeg")

        if len(hands) == 2:
            # print(detector.fingersUp(hands[0]), detector.fingersUp(hands[1]))
            if detector.fingersUp(hands[0]) == [1, 1, 0, 0, 0] and \
                    detector.fingersUp(hands[1]) == [1, 1, 0, 0, 0]:
                # print("Zoom Gesture")
                lmList1 = hands[0]["lmList"]
                lmList2 = hands[1]["lmList"]
                # point 8 is the tip of the index finger
                if startDist is None:
                    # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                    length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

                    startDist = length

                # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)
                length, info, img = detector.findDistance(hands[0]["center"], hands[1]["center"], img)

                scale = int((length - startDist) // 2)
                cx, cy = info[4:]
                print(scale)
        else:
            startDist = None

        try:
            h1, w1, _ = img1.shape
            newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
            img1 = cv2.resize(img1, (newW, newH))

            img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1
        except:
            pass

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        key = cv2.waitKey(33)
        if key != -1:
            print(key)

        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    return redirect('/')
