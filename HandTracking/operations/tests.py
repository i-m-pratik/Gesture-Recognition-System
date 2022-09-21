from django.test import TestCase
from django.shortcuts import render
from django.shortcuts import redirect
import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import os

# Create your tests here.
class DragImg():
    def __init__(self, path, posOrigin, imgType):

        self.posOrigin = posOrigin
        self.imgType = imgType
        self.path = path

        if self.imgType == 'png':
            self.img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        else:
            self.img = cv2.imread(self.path)

        # self.img = cv2.resize(self.img, (0,0),None,0.4,0.4)

        self.size = self.img.shape[:2]

    def update(self, cursor):
        ox, oy = self.posOrigin
        h, w = self.size

        # Check if in region
        if ox < cursor[0] < ox + w and oy < cursor[1] < oy + h:
            self.posOrigin = cursor[0] - w // 2, cursor[1] - h // 2



def DragDrop(request):
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.8)
    path = r"C:\Users\Win10\Desktop\be Projects\Pycharm\django\GRSDjangoUpdated\HandTracking\operations\ImagesPNG"
    myList = os.listdir(path)
    print(myList)

    listImg = []
    for x, pathImg in enumerate(myList):
        if 'png' in pathImg:
            imgType = 'png'
        else:
            imgType = 'jpg'
        listImg.append(DragImg(f'{path}/{pathImg}', [50 + x * 300, 100], imgType))

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img, flipType=False)
        print(hands, "hands")

        if hands:
            print("len1")
            lmList = hands[0]['lmList']
            print("len2")
            # Check if clicked
            length, info, img = detector.findDistance(lmList[8], lmList[12], img)
            print(length, "len")
            print(info, "info")
            print(img, "img")
            if length < 75:
                print(length, "lenu30")
                cursor = lmList[8]
                for imgObject in listImg:
                    imgObject.update(cursor)

        try:

            for imgObject in listImg:

                # Draw for JPG image
                h, w = imgObject.size
                ox, oy = imgObject.posOrigin
                if imgObject.imgType == "png":
                    # Draw for PNG Images
                    img = cvzone.overlayPNG(img, imgObject.img, [ox, oy])
                else:
                    img[oy:oy + h, ox:ox + w] = imgObject.img

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