import cv2
import numpy as np
import time
import HandTrackingModule as htm
import autopy
#########################
wCam, hCam = 640, 480
frameR = 100
smoothening = 7
clickThreshold = 0.3   # seconds - pinch longer than this becomes a drag instead of a click
#########################
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Drag state
dragging = False
pinchStartTime = 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1, y1, x2, y2)
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        # Moving Mode: only index finger up
        if fingers[1] == 1 and fingers[2] == 0:
            # if we were dragging and switched back to plain move, release the button
            if dragging:
                autopy.mouse.toggle(None, False)
                dragging = False
            pinchStartTime = 0

            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Click / Drag Mode: index + middle fingers up
        elif fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)

            if length < 40:
                # fingers pinched together
                if pinchStartTime == 0:
                    pinchStartTime = time.time()
                heldTime = time.time() - pinchStartTime

                # keep tracking cursor position while pinched (needed for dragging)
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                if not dragging and heldTime > clickThreshold:
                    # pinch held long enough -> start dragging
                    autopy.mouse.toggle(None, True)
                    dragging = True

                if dragging:
                    autopy.mouse.move(wScr - clocX, clocY)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 0, 255), cv2.FILLED)  # red = dragging
                else:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)  # green = about to click

                plocX, plocY = clocX, clocY

            else:
                # fingers just moved apart -> decide click vs end-of-drag
                if pinchStartTime != 0:
                    heldTime = time.time() - pinchStartTime
                    if dragging:
                        autopy.mouse.toggle(None, False)
                        dragging = False
                    elif heldTime <= clickThreshold:
                        autopy.mouse.click()
                pinchStartTime = 0

        else:
            # no relevant gesture - make sure nothing is left held down
            if dragging:
                autopy.mouse.toggle(None, False)
                dragging = False
            pinchStartTime = 0

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
